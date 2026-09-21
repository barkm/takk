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
    for (let k = 0; k < count; k++)
      landmarks.set([points[k].x, points[k].y, points[k].z], (first + k) * 3);
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

/** Draw one frame's landmarks over the camera image. */
export function draw(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  landmarks: Float32Array,
  edges: Edges,
): void {
  ((canvas.width = video.videoWidth), (canvas.height = video.videoHeight));
  const context = canvas.getContext("2d");
  if (!context) return;
  context.lineWidth = 3;
  for (const [group, pairs] of Object.entries(edges)) {
    context.strokeStyle = COLORS[group] ?? "#a0a6b2";
    context.beginPath();
    for (const [a, b] of pairs) {
      if (Number.isNaN(landmarks[3 * a]) || Number.isNaN(landmarks[3 * b]))
        continue;
      context.moveTo(
        landmarks[3 * a] * canvas.width,
        landmarks[3 * a + 1] * canvas.height,
      );
      context.lineTo(
        landmarks[3 * b] * canvas.width,
        landmarks[3 * b + 1] * canvas.height,
      );
    }
    context.stroke();
  }
}

export type Edges = Record<string, [number, number][]>;

const COLORS: Record<string, string> = {
  left_hand: "#60a5fa",
  right_hand: "#f87171",
  upper_body: "#a0a6b2",
  lips: "#f472b6",
};
