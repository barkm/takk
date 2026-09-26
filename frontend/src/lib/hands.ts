// Seeing when a sign begins and ends, from whether the hands are tracked at all. A search by signing
// has no spoken words to key on the way an attempt does (`voice.ts`), so the hands are what starts
// and ends it: nothing is pressed here either (user, 2026-09-24).
//
// A reading is one tracked frame. The hands are there when either hand's landmarks were detected,
// which the tracker writes as NaN when they were not — no distances and no thresholds, because a
// hand that is in the picture at all is a hand that has been raised to sign with.
import { N_LANDMARKS } from "$lib/landmarks";

// LANDMARK_SLICES in landmarks.py: the first point of each hand, which stands for the whole hand.
const LEFT = 468;
const RIGHT = 522;

export type Phase = "away" | "signing" | "done";

/** What has been seen so far: the phase, and how many readings in a row have had hands or not. */
export type Eyes = { phase: Phase; seen: number; gone: number };

export const SETTINGS = {
  seen: 3, // readings with hands that start the recording (150 ms, longer than a wave past the lens)
  gone: 14, // readings without them that end it (0.7 s, longer than a hand leaving the tracker briefly)
};

export const watching = (): Eyes => ({ phase: "away", seen: 0, gone: 0 });

/** Whether either hand was tracked in this frame. */
export function hands(landmarks: Float32Array | null): boolean {
  if (!landmarks || landmarks.length !== N_LANDMARKS * 3) return false;
  return !Number.isNaN(landmarks[LEFT * 3]) || !Number.isNaN(landmarks[RIGHT * 3]);
}

/** What one more reading makes of it. As with the voice, the two directions differ: once the hands
 * are up it takes a longer absence to count as finished, so a hand the tracker loses for a frame or
 * two in the middle of a sign does not cut the recording short. */
export function see(eyes: Eyes, present: boolean, settings = SETTINGS): Eyes {
  if (eyes.phase === "away") {
    const seen = present ? eyes.seen + 1 : 0;
    return { ...eyes, seen, phase: seen >= settings.seen ? "signing" : "away" };
  }
  if (eyes.phase === "signing") {
    const gone = present ? 0 : eyes.gone + 1;
    return { ...eyes, gone, phase: gone >= settings.gone ? "done" : "signing" };
  }
  return eyes; // done is the caller's to act on and clear
}

/** The signs of one recording made in silence (`speaks` in `about.ts`), each marked by the hands
 * going up and coming down again: `cuts` holds a start and an end per sign, in seconds on the clock
 * of `now`, at the first and the last reading that had hands. */
export type Signs = { eyes: Eyes; cuts: number[] };

export const following = (): Signs => ({ eyes: watching(), cuts: [] });

/** One more reading, taken at `now`, `tick` seconds after the one before it. */
export function follow(signs: Signs, present: boolean, now: number, tick: number, settings = SETTINGS): Signs {
  const eyes = see(signs.eyes, present, settings);
  if (signs.eyes.phase === "away" && eyes.phase === "signing")
    return { eyes, cuts: [...signs.cuts, now - (settings.seen - 1) * tick] };
  if (eyes.phase === "done") return { eyes: watching(), cuts: [...signs.cuts, now - settings.gone * tick] };
  return { eyes, cuts: signs.cuts };
}
