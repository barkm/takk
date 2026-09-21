import { describe, expect, it } from "vitest";

import { hear, listening, SETTINGS, type Ears } from "$lib/voice";

/** Feed `levels` in, one reading at a time. */
function heard(ears: Ears, ...levels: number[]): Ears {
  return levels.reduce((so_far, level) => hear(so_far, level), ears);
}

const quiet = (n: number) => Array(n).fill(0.002);
const speech = (n: number) => Array(n).fill(0.08);

/** Calibrated on a quiet room, so armed and ready for a word. */
const armed = () => heard(listening(), ...quiet(SETTINGS.calibrate));

describe("hear", () => {
  it("measures the room before it listens for a word", () => {
    const ears = armed();
    expect(ears.phase).toBe("armed");
    expect(ears.threshold).toBeCloseTo(0.006); // 0.002 * over, above the room and far under a voice
  });

  it("puts the threshold above a noisy room, and never under `least`", () => {
    expect(heard(listening(), ...Array(SETTINGS.calibrate).fill(0.02)).threshold).toBeCloseTo(0.06);
    expect(heard(listening(), ...Array(SETTINGS.calibrate).fill(0)).threshold).toBe(SETTINGS.least);
  });

  it("starts the attempt once someone has been speaking for a moment", () => {
    expect(heard(armed(), ...speech(SETTINGS.loud)).phase).toBe("speaking");
  });

  it("is not started by a single knock", () => {
    // a click is one reading loud, so the run of them is what tells a voice from a door
    const ears = heard(armed(), ...speech(SETTINGS.loud - 1), 0.002, 0.002);
    expect(ears.phase).toBe("armed");
    expect(ears.loud).toBe(0);
  });

  it("ends the attempt after the silence that follows the last word", () => {
    const speaking = heard(armed(), ...speech(SETTINGS.loud));
    expect(heard(speaking, ...quiet(SETTINGS.quiet - 1)).phase).toBe("speaking");
    expect(heard(speaking, ...quiet(SETTINGS.quiet)).phase).toBe("done");
  });

  it("is not ended by a breath between words", () => {
    const speaking = heard(armed(), ...speech(SETTINGS.loud));
    const paused = heard(speaking, ...quiet(SETTINGS.quiet - 1), 0.08, ...quiet(SETTINGS.quiet - 1));
    expect(paused.phase).toBe("speaking"); // the count starts over, so two short pauses are not one long one
  });

  it("keeps speaking through a syllable quieter than the threshold that started it", () => {
    // `under` is why: a trailing consonant sits below the threshold but well above the room
    const speaking = heard(armed(), ...speech(SETTINGS.loud));
    expect(heard(speaking, ...Array(SETTINGS.quiet).fill(speaking.threshold * 0.8)).phase).toBe("speaking");
  });
});
