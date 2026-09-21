import { describe, expect, it } from "vitest";

import { framing, FRAMING, NARROW, settle, settling, WIDE, type Settling } from "$lib/framing";
import { N_LANDMARKS } from "$lib/landmarks";

const POSE = 489;
const NOSE = POSE;
const SHOULDERS = [POSE + 11, POSE + 12];

/** One tracked frame: a signer whose shoulders are `width` of the frame wide, centred on `centre`,
 * with the head at `head` and the hands up or down. */
function frame({ width = 0.3, centre = 0.5, head = 0.25, hands = false } = {}): Float32Array {
  const landmarks = new Float32Array(N_LANDMARKS * 3).fill(NaN);
  const put = (point: number, x: number, y: number) => landmarks.set([x, y, 0], 3 * point);
  put(SHOULDERS[0], centre + width / 2, 0.6);
  put(SHOULDERS[1], centre - width / 2, 0.6);
  put(NOSE, centre, head);
  if (hands) [468, 522].forEach((wrist) => put(wrist, centre, 0.5));
  return landmarks;
}

describe("framing", () => {
  it("says nothing about a signer sitting square in the frame", () => {
    expect(framing(frame())).toBe("");
    expect(framing(frame({ hands: true }), true)).toBe("");
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

  it("asks for the hands only while a sign is being made", () => {
    expect(framing(frame({ hands: false }))).toBe("");
    expect(framing(frame({ hands: false }), true)).toBe(FRAMING.hands);
  });
});

describe("settle", () => {
  const hold = (note: string, times: number, from = settling()) =>
    Array.from({ length: times }).reduce<Settling>((state) => settle(state, note, 3), from);

  it("shows a complaint only once it has held", () => {
    expect(hold(FRAMING.hands, 2).shown).toBe("");
    expect(hold(FRAMING.hands, 3).shown).toBe(FRAMING.hands);
  });

  it("clears a shown complaint as soon as the framing is right", () => {
    expect(settle(hold(FRAMING.hands, 3), "", 3)).toEqual(settling());
  });

  it("keeps showing the old complaint while a new one settles", () => {
    const shown = hold(FRAMING.hands, 3);

    expect(settle(shown, FRAMING.near, 3).shown).toBe(FRAMING.hands);
    expect(hold(FRAMING.near, 3, shown).shown).toBe(FRAMING.near);
  });
});
