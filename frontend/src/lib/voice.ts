// Hearing when an attempt begins and ends, from how loud the microphone is (`Tracker.level`).
//
// A reading is the root mean square of the newest sound, which separates a quiet room from a
// speaking voice by two orders of magnitude — crude, but enough, because a wrong guess cannot
// produce a wrong score: a window opened by a slammed door has no spoken words in it, so forced
// alignment scores them below MIN_WORD_SCORE and the server refuses the attempt. The cost of a
// false start is one wasted window, which is why there is no voice detector here.
//
// Microphones differ too much for an absolute threshold, so the quiet room is measured first and
// the threshold set above it.

export type Phase = "calibrating" | "armed" | "speaking" | "done";

/** What has been heard so far: the phase, the readings of the quiet room until there are enough of
 * them, the threshold measured from those, and how many readings in a row have been loud or quiet. */
export type Ears = { phase: Phase; floor: number[]; threshold: number; loud: number; quiet: number };

export const SETTINGS = {
  calibrate: 14, // readings of the quiet room taken before arming, at 50 ms each
  over: 3, // times the room's noise floor that counts as someone speaking
  under: 0.6, // the share of the threshold a reading must fall under to count as quiet again
  least: 0.01, // a threshold no lower than this, for a microphone whose floor is near silence
  loud: 3, // readings above the threshold that start the attempt (150 ms, longer than a click)
  quiet: 24, // readings under it that end it (1.2 s, longer than a breath between words)
};

export const listening = (): Ears => ({ phase: "calibrating", floor: [], threshold: 0, loud: 0, quiet: 0 });

/** What one more reading makes of it. The thresholds differ in each direction on purpose: once
 * someone is speaking it takes a quieter stretch to count as finished, so a soft last syllable or a
 * breath between words does not cut the attempt short. */
export function hear(ears: Ears, level: number, settings = SETTINGS): Ears {
  if (ears.phase === "calibrating") {
    const floor = [...ears.floor, level];
    if (floor.length < settings.calibrate) return { ...ears, floor };
    const middle = [...floor].sort((a, b) => a - b)[Math.floor(floor.length / 2)];
    return { ...ears, floor, phase: "armed", threshold: Math.max(middle * settings.over, settings.least) };
  }
  if (ears.phase === "armed") {
    const loud = level > ears.threshold ? ears.loud + 1 : 0;
    return { ...ears, loud, phase: loud >= settings.loud ? "speaking" : "armed" };
  }
  const quiet = level < ears.threshold * settings.under ? ears.quiet + 1 : 0;
  return { ...ears, quiet, phase: quiet >= settings.quiet ? "done" : ears.phase };
}
