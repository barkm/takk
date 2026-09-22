// Landmarks are extracted here, live, while the camera runs: MediaPipe's HolisticLandmarker in VIDEO
// mode on the GPU (the CPU if the GPU is unavailable), with the same version and model as
// extraction.py. Only the landmarks of a recording are sent to the server, never the video.
import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision";

import { api } from "$lib/api";
import { draw, layout, type Edges, type Frame } from "$lib/landmarks";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

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

  private constructor(
    private video: HTMLVideoElement,
    private canvas: HTMLCanvasElement,
    private landmarker: HolisticLandmarker,
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
    const files = await FilesetResolver.forVisionTasks(WASM);
    const create = (delegate: "GPU" | "CPU") =>
      HolisticLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: api("/api/model"), delegate }, runningMode: "VIDEO" });
    let delegate: "GPU" | "CPU" = "GPU";
    let landmarker: HolisticLandmarker;
    try {
      landmarker = await create(delegate);
    } catch {
      delegate = "CPU";
      landmarker = await create(delegate);
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
    let last = -1;
    const next = (now: number, metadata: VideoFrameCallbackMetadata) => {
      if (this.done) return;
      const time = metadata.captureTime ?? now;
      last = Math.max(last + 1, Math.round(time)); // VIDEO mode needs increasing timestamps
      const landmarks = layout(this.landmarker.detectForVideo(this.video, last));
      this.latest = landmarks;
      draw(this.canvas, { width: this.video.videoWidth, height: this.video.videoHeight }, landmarks, this.edges);
      if (this.recorded) this.recorded.push({ time: time / 1000, landmarks });
      this.video.requestVideoFrameCallback(next);
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
    this.landmarker.close();
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
