// The shape of a pass of Ord: how many words it holds and how often each of them comes back. They
// are fixed: asking the learner for them would ask a decision of them for no gain.
import type { SignWord } from "$lib/api";

export const WORDS = 5;
export const REPEATS = 2;

/** The words of one pass of Ord, in the order they are signed: every word once, then every word
 * again, `repeats` times over. A word comes back with the others in between
 * rather than straight after itself, since signing it twice in a row tests reading the clip and not
 * remembering the sign. */
export function pass(words: SignWord[], repeats: number): SignWord[] {
  return Array.from({ length: repeats }, () => words).flat();
}
