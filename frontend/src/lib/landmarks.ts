// The common landmark layout of landmarks.py: one frame is N_LANDMARKS points of x, y, z, NaN where
// a part was not detected. Everything the page sends to the server is in this layout.
import type { HolisticLandmarkerResult } from "@mediapipe/tasks-vision";

export const N_LANDMARKS = 543;

// LANDMARK_SLICES in landmarks.py: [result field, first index, number of points]
const PARTS = [
  ["faceLandmarks", 0, 468], // the face model's 10 iris points after these are dropped
  ["leftHandLandmarks", 468, 21],
  ["poseLandmarks", 489, 33],
  ["rightHandLandmarks", 522, 21],
] as const;

export type Frame = { time: number; landmarks: Float32Array };

/** The landmarks of one result in the common layout, NaN where a part was not detected. */
export function layout(result: HolisticLandmarkerResult): Float32Array {
  const landmarks = new Float32Array(N_LANDMARKS * 3).fill(NaN);
  for (const [field, first, count] of PARTS) {
    const points = result[field][0];
    if (!points) continue;
    for (let k = 0; k < count; k++) landmarks.set([points[k].x, points[k].y, points[k].z], (first + k) * 3);
  }
  return landmarks;
}

/** The landmarks at the preparation's constant frame rate: at every 1 / fps seconds the latest tracked
 * frame, as the collection app's transcode picks video frames. */
export function resample(frames: Frame[], fps: number): Float32Array {
  const start = frames[0].time;
  const n = Math.floor((frames.at(-1)!.time - start) * fps) + 1;
  const landmarks = new Float32Array(n * N_LANDMARKS * 3);
  for (let i = 0, j = 0; i < n; i++) {
    while (j + 1 < frames.length && frames[j + 1].time <= start + i / fps) j++;
    landmarks.set(frames[j].landmarks, i * N_LANDMARKS * 3);
  }
  return landmarks;
}

/** Draw one frame's landmarks, at `size` pixels: over the camera image while tracking, and on its own
 * when a recording is replayed, since nothing of the camera's picture is kept. */
export function draw(canvas: HTMLCanvasElement, size: { width: number; height: number }, landmarks: Float32Array, edges: Edges): void {
  (canvas.width = size.width), (canvas.height = size.height);
  const context = canvas.getContext("2d");
  if (!context) return;
  // The canvas has the stream's own size and is shown scaled down to cover the view, so widths are
  // set in the pixels of the view: the same weight whichever box the picture sits in.
  const scale = Math.max(canvas.clientWidth / canvas.width, canvas.clientHeight / canvas.height) || 0.25;
  (context.lineCap = "round"), (context.lineJoin = "round");
  for (const [group, pairs] of Object.entries(edges)) {
    const style = STYLES[group] ?? STYLES.upper_body;
    (context.strokeStyle = style.color), (context.lineWidth = style.width / scale);
    context.beginPath();
    for (const [a, b] of pairs) {
      if (Number.isNaN(landmarks[3 * a]) || Number.isNaN(landmarks[3 * b])) continue;
      context.moveTo(landmarks[3 * a] * canvas.width, landmarks[3 * a + 1] * canvas.height);
      context.lineTo(landmarks[3 * b] * canvas.width, landmarks[3 * b + 1] * canvas.height);
    }
    context.stroke();
  }
}

/** A One Euro filter on every coordinate, for the drawn landmarks only: it smooths hard while a point
 * is still, which takes out the tracker's jitter, and less the faster it moves, so a sign does not
 * trail behind. What is recorded and scored stays raw, as extraction.py leaves it. A point that is
 * lost starts afresh where it is found again. Coordinates are fractions of the picture per second. */
export class Smoother {
  private value = new Float32Array(N_LANDMARKS * 3).fill(NaN);
  private speed = new Float32Array(N_LANDMARKS * 3);
  private time = -Infinity;

  constructor(
    private minCutoff = 1, // Hz: how still a still point is held
    private beta = 10, // how fast the cutoff rises with speed, i.e. how little a moving point lags
  ) {}

  /** A smoothed copy of one frame's landmarks, tracked at `time` seconds. */
  smooth(landmarks: Float32Array, time: number): Float32Array {
    const dt = time - this.time;
    this.time = time;
    const alpha = (cutoff: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
    const alphaSpeed = alpha(1); // the speed itself is smoothed at a fixed 1 Hz
    for (let i = 0; i < landmarks.length; i++) {
      const x = landmarks[i];
      if (Number.isNaN(x) || Number.isNaN(this.value[i]) || !(dt > 0)) {
        (this.value[i] = x), (this.speed[i] = 0);
        continue;
      }
      this.speed[i] += alphaSpeed * ((x - this.value[i]) / dt - this.speed[i]);
      this.value[i] += alpha(this.minCutoff + this.beta * Math.abs(this.speed[i])) * (x - this.value[i]);
    }
    return this.value.slice();
  }
}

export type Edges = Record<string, [number, number][]>;

// The hands are what a sign is made with, so they are drawn bright and heavy and the rest recedes
// (user, 2026-09-26). No red: red means a miss and a recording running. Widths in pixels of the view.
const STYLES: Record<string, { color: string; width: number }> = {
  left_hand: { color: "#ffb000", width: 2.5 },
  right_hand: { color: "#4dabf7", width: 2.5 },
  upper_body: { color: "rgba(255, 255, 255, 0.55)", width: 1.75 },
  lips: { color: "rgba(255, 255, 255, 0.4)", width: 1 },
};
