import { describe, expect, it } from "vitest";

import { N_LANDMARKS, Smoother, layout, resample, type Frame } from "$lib/landmarks";

const point = (value: number) => ({ x: value, y: value, z: value });

describe("layout", () => {
  it("puts each part where landmarks.py expects it, and NaN where nothing was detected", () => {
    const result = {
      faceLandmarks: [Array.from({ length: 468 }, () => point(0.1))],
      leftHandLandmarks: [Array.from({ length: 21 }, () => point(0.2))],
      poseLandmarks: [Array.from({ length: 33 }, () => point(0.3))],
      rightHandLandmarks: [], // not detected in this frame
    };
    const frame = layout(result as never);
    expect(frame.length).toBe(N_LANDMARKS * 3);
    expect(frame[0]).toBeCloseTo(0.1); // the face starts at 0
    expect(frame[468 * 3]).toBeCloseTo(0.2); // the left hand at 468
    expect(frame[489 * 3]).toBeCloseTo(0.3); // the pose at 489
    expect(frame[522 * 3]).toBeNaN(); // the right hand at 522
  });
});

describe("resample", () => {
  const frames = (times: number[]): Frame[] =>
    times.map((time, i) => ({ time, landmarks: new Float32Array(N_LANDMARKS * 3).fill(i) }));

  it("takes the latest tracked frame at every step of the preparation's frame rate", () => {
    const landmarks = resample(frames([0, 0.5, 1.0]), 2); // 2 fps over 1 s: steps at 0, 0.5, 1.0
    expect(landmarks.length).toBe(3 * N_LANDMARKS * 3);
    expect([0, 1, 2].map((i) => landmarks[i * N_LANDMARKS * 3])).toEqual([0, 1, 2]);
  });

  it("repeats the latest frame when tracking is slower than the frame rate", () => {
    const landmarks = resample(frames([0, 1.0]), 2); // one tracked frame per second, resampled to 2 fps
    expect([0, 1, 2].map((i) => landmarks[i * N_LANDMARKS * 3])).toEqual([0, 0, 1]);
  });
});

describe("Smoother", () => {
  const frame = (value: number) => new Float32Array(N_LANDMARKS * 3).fill(value);

  it("holds a still point steadier than the tracker does, and follows it once it moves", () => {
    const smoother = new Smoother();
    let out = smoother.smooth(frame(0.5), 0);
    for (let i = 1; i <= 30; i++) out = smoother.smooth(frame(0.5 + (i % 2 ? 0.01 : -0.01)), i / 30);
    expect(Math.abs(out[0] - 0.5)).toBeLessThan(0.005); // jitter of ±0.01 held to under half
    for (let i = 31; i <= 60; i++) out = smoother.smooth(frame(0.8), i / 30);
    expect(out[0]).toBeCloseTo(0.8, 2); // a jump is reached within a second
  });

  it("starts a point afresh where it is found again", () => {
    const smoother = new Smoother();
    smoother.smooth(frame(0.2), 0);
    expect(smoother.smooth(frame(NaN), 1 / 30)[0]).toBeNaN();
    expect(smoother.smooth(frame(0.9), 2 / 30)[0]).toBeCloseTo(0.9);
  });
});
