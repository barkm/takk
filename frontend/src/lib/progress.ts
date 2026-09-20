// What the learner has practised, kept in the browser (step 8 of ROADMAP-takk.md): no accounts and
// no state on the server for as long as that holds. One Leitner box per sign: an accepted attempt
// moves it up a box, a rejected one back to the first, and the box says how long until it is due.
import type { Pack, PackWord } from "$lib/api";

const KEY = "takk.progress";
const CHOSEN = "takk.packs";
const DAY = 24 * 60 * 60 * 1000;
/** Days until a sign in each box is due again. The first box is practised within the same session. */
const DAYS = [0, 1, 3, 7, 21];

/** A sign's box (1 and up) and when it is due again, in milliseconds since the epoch. */
export type Learned = { box: number; due: number };
export type Progress = Record<string, Learned>;

export function load(): Progress {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "{}");
  } catch {
    return {}; // a corrupt or unreadable store is a fresh start, never a broken page
  }
}

export function save(progress: Progress) {
  localStorage.setItem(KEY, JSON.stringify(progress));
}

/** The progress after an attempt of `sign` was accepted or rejected. */
export function record(progress: Progress, sign: string, correct: boolean, now = Date.now()): Progress {
  const box = correct ? Math.min((progress[sign]?.box ?? 0) + 1, DAYS.length) : 1;
  return { ...progress, [sign]: { box, due: now + DAYS[box - 1] * DAY } };
}

/** The packs the learner practises, the starter packs until they choose otherwise. */
export function loadChosen(packs: Pack[]): string[] {
  const starters = packs.filter((pack) => pack.kind === "pack").map((pack) => pack.name);
  let chosen: string[] = starters;
  try {
    chosen = JSON.parse(localStorage.getItem(CHOSEN) ?? "null") ?? starters;
  } catch {
    chosen = starters; // an unreadable store is a fresh start, as in `load`
  }
  return chosen.filter((name) => packs.some((pack) => pack.name === name)); // a pack the lexicon dropped
}

export function saveChosen(chosen: string[]) {
  localStorage.setItem(CHOSEN, JSON.stringify(chosen));
}

/** The words of the chosen packs, each sign once however many packs it is in. */
export function chosenWords(packs: Pack[], chosen: string[]): PackWord[] {
  const words = new Map<string, PackWord>();
  for (const pack of packs.filter((pack) => chosen.includes(pack.name)))
    for (const word of pack.words) words.set(word.sign, words.get(word.sign) ?? word);
  return [...words.values()];
}

/** The signs of today's session: the due ones first, then words never practised, at most `size`. */
export function session(words: PackWord[], progress: Progress, size = 5, now = Date.now()): PackWord[] {
  const due = words.filter((word) => progress[word.sign] && progress[word.sign].due <= now);
  due.sort((a, b) => progress[a.sign].due - progress[b.sign].due);
  const fresh = words.filter((word) => !progress[word.sign]);
  return [...due, ...fresh].slice(0, size);
}
