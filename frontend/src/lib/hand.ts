// Which hand the learner signs with, asked once and kept in the browser: it is what a recording is
// mirrored by, so every attempt needs it and nothing else does. Asked on Om dig the first time the
// app is opened (step 23 of ROADMAP-takk.md), so no camera screen carries a switch for it.
export type Hand = "left" | "right";

const KEY = "takk.handedness";

/** The hand the learner signs with, or null until they have been asked. */
export function hand(): Hand | null {
  try {
    const kept = localStorage.getItem(KEY);
    return kept === "left" || kept === "right" ? kept : null;
  } catch {
    return null; // an unreadable store asks again, as the progress store starts over
  }
}

export function setHand(which: Hand) {
  localStorage.setItem(KEY, which);
}
