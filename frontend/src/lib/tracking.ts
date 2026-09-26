// Landmarks are extracted here, live, while the camera runs: MediaPipe's HolisticLandmarker in VIDEO
// mode on the GPU (the CPU if the GPU is unavailable), with the same version and model as
// extraction.py, in a worker of its own (`landmarker.worker.ts`). Only the landmarks of a recording
// are sent to the server, never the video.
import { Smoother, draw, type Edges, type Frame } from "$lib/landmarks";

/** The landmarker, loaded once for the whole visit and shared by the pages' cameras, so moving
 * between pages neither loads it again nor has two loading at once. */
let loaded: Promise<Worker> | null = null;
let last = -1; // the timestamp of the frame sent last, which VIDEO mode needs to increase across pages

function load(): Promise<Worker> {
  const landmarker = new Worker(new URL("./landmarker.worker.ts", import.meta.url), { type: "module" });
  return new Promise((resolve, reject) => {
    landmarker.onmessage = ({ data }) => (data.ready ? resolve(landmarker) : reject(new Error(data.error)));
    landmarker.onerror = (event) => reject(new Error(event.message));
  }).catch((error) => {
    (loaded = null), landmarker.terminate(); // the next camera tries again
    throw error;
  }) as Promise<Worker>;
}

export type Recording = { frames: Frame[]; audio: Blob | null; audioStart: number };

export class Tracker {
  /** Whether the microphone is open. A sentence of several signs is split by the words the signer
   * speaks, so without it only one sign at a time can be practised. */
  readonly hasAudio: boolean;

  /** Which landmarks are joined by a line, from the lexicon. The camera starts before the lexicon
   * has arrived, so this is set once it has rather than passed in. */
  edges: Edges = {};

  private recorded: Frame[] | null = null;
  private done = false; // set by `stop`, which the frame loop reads to let itself end
  /** The landmarks of the frame tracked last, which is what the framing feedback reads. */
  latest: Float32Array | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private stopped: Promise<Blob> | null = null;
  private audioStart = 0;
  private context: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private samples = new Float32Array(0);
  private smoother = new Smoother();
  private listener = (_: MessageEvent) => {};

  private constructor(
    private video: HTMLVideoElement,
    private canvas: HTMLCanvasElement,
    private landmarker: Worker,
    hasAudio: boolean,
  ) {
    this.hasAudio = hasAudio;
  }

  /** Open the camera, load the landmarker and start tracking. The microphone is what splits a
   * sentence into its signs, but a single sign is the whole recording, so a refused microphone
   * leaves that much working rather than being an error. */
  static async start(video: HTMLVideoElement, canvas: HTMLCanvasElement): Promise<Tracker> {
    const constraints = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
    const stream = await navigator.mediaDevices
      .getUserMedia({ video: constraints, audio: true })
      .catch(() => navigator.mediaDevices.getUserMedia({ video: constraints }));
    video.srcObject = stream;
    let landmarker: Worker;
    try {
      landmarker = await (loaded ??= load());
    } catch (error) {
      stream.getTracks().forEach((track) => track.stop());
      throw error;
    }
    const tracker = new Tracker(video, canvas, landmarker, stream.getAudioTracks().length > 0);
    tracker.listen(stream);
    tracker.track();
    return tracker;
  }

  /** Watch how loud the microphone is, which is what starts and stops a recording. The analyser
   * keeps the newest samples in a buffer that `level` reads whenever it is asked, so nothing runs
   * between reads and the frame loop can poll it. */
  private listen(stream: MediaStream): void {
    if (!this.hasAudio) return;
    this.context = new AudioContext();
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 1024; // about 21 ms at 48 kHz, shorter than any word
    this.context.createMediaStreamSource(stream).connect(this.analyser);
    this.samples = new Float32Array(this.analyser.fftSize);
  }

  /** How loud the last ~21 ms were, as the root mean square of the waveform: near 0.002 in a quiet
   * room and above 0.05 while someone speaks. Zero without a microphone. Absolute values mean little
   * across microphones, so the caller compares it against a floor it measured itself. */
  get level(): number {
    if (!this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this.samples);
    let square = 0;
    for (const sample of this.samples) square += sample * sample;
    return Math.sqrt(square / this.samples.length);
  }

  /** Browsers may open an audio context suspended until the page has been clicked. */
  resume(): void {
    if (this.context?.state === "suspended") void this.context.resume().catch(() => {}); // until the page is clicked
  }

  /** Run the landmarker on every camera frame it can keep up with; frames that arrive while it is
   * busy are skipped. While recording, each result is kept with its frame's capture time. */
  private track(): void {
    let busy = false;
    let time = 0; // the capture time of the frame the landmarker is working on
    this.listener = ({ data: { landmarks } }: MessageEvent<{ landmarks: Float32Array }>) => {
      busy = false;
      if (this.done) return;
      this.latest = landmarks;
      const size = { width: this.video.videoWidth, height: this.video.videoHeight };
      draw(this.canvas, size, this.smoother.smooth(landmarks, last / 1000), this.edges);
      if (this.recorded) this.recorded.push({ time: time / 1000, landmarks });
    };
    this.landmarker.addEventListener("message", this.listener);
    const next = async (now: number, metadata: VideoFrameCallbackMetadata) => {
      if (this.done) return;
      this.video.requestVideoFrameCallback(next);
      if (busy) return;
      busy = true;
      const captured = metadata.captureTime ?? now;
      const frame = await createImageBitmap(this.video).catch(() => null); // none once the camera is closed
      if (!frame || this.done) return void ((busy = false), frame?.close());
      (time = captured), (last = Math.max(last + 1, Math.round(captured))); // VIDEO mode needs increasing timestamps
      this.landmarker.postMessage({ frame, time: last }, [frame]);
    };
    this.video.requestVideoFrameCallback(next);
  }

  get recording(): boolean {
    return this.recorded !== null;
  }

  /** Close the camera and the microphone and stop tracking, which a page does when it is done with
   * them: nothing else stops the stream, so an unmounted page would leave the camera light on and the
   * next one would open a second stream beside it. */
  stop(): void {
    this.done = true;
    (this.video.srcObject as MediaStream | null)?.getTracks().forEach((track) => track.stop());
    this.video.srcObject = null;
    void this.context?.close().catch(() => {});
    this.landmarker.removeEventListener("message", this.listener);
  }

  /** Start keeping the tracked frames, and the spoken sentence alongside them in whichever container
   * the browser picks (the server decodes it with ffmpeg). The audio's start is on the same clock as
   * the frames' capture times, so their difference places the words among the frames. */
  startRecording(): void {
    this.recorded = [];
    const tracks = (this.video.srcObject as MediaStream).getAudioTracks();
    if (!tracks.length) return;
    this.chunks = [];
    this.recorder = new MediaRecorder(new MediaStream(tracks));
    this.recorder.ondataavailable = (event) => this.chunks.push(event.data);
    this.recorder.onstart = () => (this.audioStart = performance.now() / 1000);
    this.stopped = new Promise((resolve) => this.recorder && (this.recorder.onstop = () => resolve(new Blob(this.chunks))));
    this.recorder.start();
  }

  /** The recording so far, or null when nothing is being recorded (the time limit and the button can
   * both stop one). */
  async stopRecording(): Promise<Recording | null> {
    const frames = this.recorded;
    this.recorded = null;
    if (!frames) return null;
    let audio: Blob | null = null;
    if (this.recorder) {
      this.recorder.stop();
      audio = await this.stopped;
      (this.recorder = null), (this.stopped = null);
    }
    return { frames, audio, audioStart: this.audioStart };
  }
}
