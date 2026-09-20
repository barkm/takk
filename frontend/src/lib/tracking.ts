// Landmarks are extracted here, live, while the camera runs: MediaPipe's HolisticLandmarker in VIDEO
// mode on the GPU (the CPU if the GPU is unavailable), with the same version and model as
// extraction.py. Only the landmarks of a recording are sent to the server, never the video.
import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision";

import { api } from "$lib/api";
import { draw, layout, type Edges, type Frame } from "$lib/landmarks";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

export type Recording = { frames: Frame[]; audio: Blob | null; audioStart: number };

export class Tracker {
  /** Which delegate the landmarker runs on, to show alongside the frame rate. */
  readonly delegate: "GPU" | "CPU";
  /** Whether the sentence can be split by the spoken words, which needs the microphone. */
  readonly hasAudio: boolean;

  private recorded: Frame[] | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private stopped: Promise<Blob> | null = null;
  private audioStart = 0;

  private constructor(
    private video: HTMLVideoElement,
    private canvas: HTMLCanvasElement,
    private landmarker: HolisticLandmarker,
    private edges: Edges,
    private onRate: (fps: number) => void,
    delegate: "GPU" | "CPU",
    hasAudio: boolean,
  ) {
    (this.delegate = delegate), (this.hasAudio = hasAudio);
  }

  /** Open the camera, load the landmarker and start tracking. The microphone is what splits a
   * sentence into its signs; without it the rests do, so a refused microphone is not an error. */
  static async start(video: HTMLVideoElement, canvas: HTMLCanvasElement, edges: Edges, onRate: (fps: number) => void): Promise<Tracker> {
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
    const tracker = new Tracker(video, canvas, landmarker, edges, onRate, delegate, stream.getAudioTracks().length > 0);
    tracker.track();
    return tracker;
  }

  /** Run the landmarker on every camera frame it can keep up with; frames that arrive while it is
   * busy are skipped. While recording, each result is kept with its frame's capture time. */
  private track(): void {
    let last = -1;
    let count = 0;
    let since = performance.now();
    const next = (now: number, metadata: VideoFrameCallbackMetadata) => {
      const time = metadata.captureTime ?? now;
      last = Math.max(last + 1, Math.round(time)); // VIDEO mode needs increasing timestamps
      const landmarks = layout(this.landmarker.detectForVideo(this.video, last));
      draw(this.canvas, this.video, landmarks, this.edges);
      if (this.recorded) this.recorded.push({ time: time / 1000, landmarks });
      count += 1;
      if (now - since > 1000) {
        this.onRate((count * 1000) / (now - since));
        (count = 0), (since = now);
      }
      this.video.requestVideoFrameCallback(next);
    };
    this.video.requestVideoFrameCallback(next);
  }

  get recording(): boolean {
    return this.recorded !== null;
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
