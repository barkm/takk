import { describe, expect, it } from "vitest";

import { hands, see, SETTINGS, watching, type Eyes } from "$lib/hands";
import { N_LANDMARKS } from "$lib/landmarks";

/** Feed `readings` in, one tracked frame at a time. */
function seen(eyes: Eyes, ...readings: boolean[]): Eyes {
  return readings.reduce((so_far, present) => see(so_far, present), eyes);
}

const up = (n: number) => Array(n).fill(true);
const away = (n: number) => Array(n).fill(false);

/** Hands raised long enough that the recording is running. */
const signing = () => seen(watching(), ...up(SETTINGS.seen));

describe("hands", () => {
  it("sees a hand when either one was tracked", () => {
    const frame = new Float32Array(N_LANDMARKS * 3).fill(NaN);
    expect(hands(frame)).toBe(false);
    frame[522 * 3] = 0.5; // the right hand
    expect(hands(frame)).toBe(true);
  });

  it("sees nothing in a frame that is not a frame", () => {
    expect(hands(null)).toBe(false);
    expect(hands(new Float32Array(3))).toBe(false);
  });
});

describe("see", () => {
  it("starts once the hands have been up for a moment", () => {
    expect(seen(watching(), ...up(SETTINGS.seen - 1)).phase).toBe("away");
    expect(signing().phase).toBe("signing");
  });

  it("ends when the hands have been gone long enough", () => {
    expect(seen(signing(), ...away(SETTINGS.gone - 1)).phase).toBe("signing");
    expect(seen(signing(), ...away(SETTINGS.gone)).phase).toBe("done");
  });

  it("keeps going when a hand is lost for a frame or two", () => {
    const flickering = seen(signing(), ...away(SETTINGS.gone - 1), true, ...away(SETTINGS.gone - 1));
    expect(flickering.phase).toBe("signing");
  });

  it("leaves what to do about a finished sign to its caller", () => {
    const done = seen(signing(), ...away(SETTINGS.gone));
    expect(seen(done, ...up(SETTINGS.seen)).phase).toBe("done");
  });
});
