// What the learner sets before a pass of Ord starts: how long it is and how often each of its words
// comes back. They are chosen on its start card (user, 2026-09-24) and kept in the browser beside the
// progress store, so a learner who practises the same way every day picks it once.
import type { SignWord } from "$lib/api";

const KEY = "takk.options";

export type Options = { words: number; repeats: number };

/** The values each option offers. A few sensible ones rather than any number at all: the page asks
 * for a tap, not for a decision about what a good number would be. */
export const WORDS = [3, 5, 10, 20];
export const REPEATS = [1, 2, 3];

/** What a learner who has set nothing practises with, and what a page holds until it has read the
 * store — which it does in an effect, as the progress store is read. */
export const DEFAULTS: Options = { words: 5, repeats: 1 };

export function load(): Options {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return DEFAULTS; // an unreadable store is the defaults, never a broken page
  }
}

export function save(chosen: Options) {
  localStorage.setItem(KEY, JSON.stringify(chosen));
}

/** The words of one pass of Ord, in the order they are signed: every word once, then every word
 * again, `repeats` times over (user, 2026-09-24). A word comes back with the others in between
 * rather than straight after itself, since signing it twice in a row tests reading the clip and not
 * remembering the sign. */
export function pass(words: SignWord[], repeats: number): SignWord[] {
  return Array.from({ length: repeats }, () => words).flat();
}
