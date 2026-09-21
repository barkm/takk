import { describe, expect, it } from "vitest";

import { framing, FRAMING, NARROW, WIDE } from "$lib/framing";
import { N_LANDMARKS } from "$lib/landmarks";

const POSE = 489;
const NOSE = POSE;
const SHOULDERS = [POSE + 11, POSE + 12];

/** One tracked frame: a signer whose shoulders are `width` of the frame wide, centred on `centre`,
 * with the head at `head`. */
function frame({ width = 0.3, centre = 0.5, head = 0.25 } = {}): Float32Array {
  const landmarks = new Float32Array(N_LANDMARKS * 3).fill(NaN);
  const put = (point: number, x: number, y: number) => landmarks.set([x, y, 0], 3 * point);
  put(SHOULDERS[0], centre + width / 2, 0.6);
  put(SHOULDERS[1], centre - width / 2, 0.6);
  put(NOSE, centre, head);
  return landmarks;
}

describe("framing", () => {
  it("says nothing about a signer sitting square in the frame", () => {
    expect(framing(frame())).toBe("");
  });

  it("asks for the shoulders before anything else", () => {
    expect(framing(new Float32Array(N_LANDMARKS * 3).fill(NaN))).toBe(FRAMING.shoulders);
    expect(framing(new Float32Array(3))).toBe(FRAMING.none);
  });

  it("says which way to move", () => {
    expect(framing(frame({ width: NARROW - 0.01 }))).toBe(FRAMING.near);
    expect(framing(frame({ width: WIDE + 0.01 }))).toBe(FRAMING.far);
    expect(framing(frame({ centre: 0.8 }))).toBe(FRAMING.centre);
    expect(framing(frame({ head: 0.02 }))).toBe(FRAMING.headroom); // no room for the signs above the head
  });
});
