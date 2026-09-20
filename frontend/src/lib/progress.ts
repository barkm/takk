// What the learner has practised, kept in the browser (step 8 of ROADMAP-takk.md): no accounts and
// no state on the server for as long as that holds. One Leitner box per sign, saying how long until
// it is due: a word moves up a box once a pass has accepted it `accepts` times without missing it,
// and back to the first box the moment it is missed.
import { word as signWord, type Pack, type PackWord } from "$lib/api";

const KEY = "takk.progress";
const CHOSEN = "takk.packs";
const SETTINGS = { size: "takk.pass-size", accepts: "takk.accepts" };
const DAY = 24 * 60 * 60 * 1000;
/** Days until a sign in each box is due again. The first box waits a day like the second: a sign that
 * was missed comes back within the same pass, so a finished pass is finished. */
export const DAYS = [1, 1, 3, 7, 21];

/** A sign's box (1 and up), when it is due again and when it was last practised, in milliseconds
 * since the epoch, and the word the learner was shown. The word is kept because a box belongs to a
 * sign class and a class has many words: `sts:spader-00016` is "svart" in Färger and "Oden" in
 * Mytologi, and what was practised is the one that was on the card. `first` is when the sign was met
 * for the first time, which is the only way to tell a new word from a missed old one: both sit in the
 * first box. Boxes written before these fields existed count as neither practised nor met today and
 * fall back to a pack's word. */
export type Learned = { box: number; due: number; seen?: number; first?: number; word?: string };

/** The learner's own numbers, until they say otherwise: `size` words in a pass, and `accepts` accepted
 * attempts of each before it is finished and moves up a box. */
export const DEFAULTS = { size: 5, accepts: 2 };
export type Setting = keyof typeof DEFAULTS;
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

/** The progress after a turn of `sign`, shown as `word`, finished the word or missed it. It is called
 * once a word's fate in the pass is settled, not on every turn: an accepted attempt that still leaves
 * the word short of its accepts changes nothing, so a word met today and left unfinished is met again
 * from the start, with its tutorial, rather than sitting half-learned in a box. */
export function record(progress: Progress, sign: string, correct: boolean, word = "", now = Date.now()): Progress {
  const box = correct ? Math.min((progress[sign]?.box ?? 0) + 1, DAYS.length) : 1;
  const learned = { box, due: now + DAYS[box - 1] * DAY, seen: now, first: progress[sign]?.first ?? now };
  return { ...progress, [sign]: word ? { ...learned, word } : learned };
}

/** How many signs were practised since midnight, which is what a day's work amounts to. */
export function practisedToday(progress: Progress, now = Date.now()): number {
  const midnight = new Date(now).setHours(0, 0, 0, 0);
  return Object.values(progress).filter((learned) => (learned.seen ?? 0) >= midnight).length;
}

/** One of the learner's numbers, their default until it is chosen. Both count things a pass cannot
 * have none of, so anything below 1 is not an answer and falls back to the default. */
export function loadSetting(name: Setting): number {
  const value = Number(localStorage.getItem(SETTINGS[name]));
  return Number.isFinite(value) && value >= 1 ? Math.floor(value) : DEFAULTS[name];
}

export function saveSetting(name: Setting, value: number) {
  localStorage.setItem(SETTINGS[name], String(Math.max(Math.floor(value), 1)));
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

/** The words of the chosen packs, each sign once however many packs it is in, taken a word at a time
 * from each pack in turn.
 *
 * The packs are read in turn rather than one after the other because a session takes its new words
 * from the front: in order, Djur's 196 words would come before Mat och dryck's first one, which at
 * five new words a day is a month of animals. Each pack keeps its own order, the most counted first,
 * and a pack that runs out drops out of the round. */
export function chosenWords(packs: Pack[], chosen: string[]): PackWord[] {
  const chosenPacks = packs.filter((pack) => chosen.includes(pack.name));
  const words = new Map<string, PackWord>();
  const longest = Math.max(0, ...chosenPacks.map((pack) => pack.words.length));
  for (let at = 0; at < longest; at++)
    for (const pack of chosenPacks) {
      const word = pack.words[at];
      if (word) words.set(word.sign, words.get(word.sign) ?? word);
    }
  return [...words.values()];
}

/** A practised sign, as the progress page shows it: soonest due first. */
export type Row = Learned & { sign: string; word: string; packs: string[] };

/** Everything practised so far, with the word to show it by and the packs that teach it under that
 * word. A sign no longer in any pack is still listed, by the word its sign is named for: it was
 * practised all the same. The packs are the word's, not the sign class's, because a class carries
 * several words: `sts:spader-00016` is in Färger as "svart", in Mat och dryck as "lakrits" and in
 * Spel as "spader", and only the first of those is where "svart" is practised. */
export function boxes(progress: Progress, packs: Pack[]): Row[] {
  const named = new Map<string, string>(); // the packs hold 14,000 words, so they are walked once
  const inPacks = new Map<string, string[]>();
  for (const pack of packs)
    for (const word of pack.words) {
      if (word.word && !named.has(word.sign)) named.set(word.sign, word.word); // the first pack names it
      const key = `${word.sign}\0${word.word ?? signWord(word.sign)}`;
      inPacks.set(key, [...(inPacks.get(key) ?? []), pack.name]);
    }
  const rows = Object.entries(progress).map(([sign, learned]) => {
    const word = learned.word ?? named.get(sign) ?? signWord(sign);
    return { ...learned, sign, word, packs: inPacks.get(`${sign}\0${word}`) ?? [] };
  });
  return rows.sort((a, b) => a.due - b.due || a.word.localeCompare(b.word, "sv"));
}

/** The signs of one pass: `size` of them, the due ones first and soonest due first, filled up with
 * words never practised when fewer than `size` are due. A pass is therefore the same length whatever
 * the boxes hold, and new words arrive only as far as the repetitions leave room for them. */
export function session(words: PackWord[], progress: Progress, size = DEFAULTS.size, now = Date.now()): PackWord[] {
  const due = words.filter((word) => progress[word.sign] && progress[word.sign].due <= now);
  due.sort((a, b) => progress[a.sign].due - progress[b.sign].due);
  const fresh = words.filter((word) => !progress[word.sign]);
  return [...due, ...fresh].slice(0, size);
}

/** The turns left in a pass after the current one, given how often each sign has been accepted in it.
 * A word short of `needed` goes back to the end of the queue, so the pass cycles until every word has
 * been accepted `needed` times, and a missed word comes back in the same pass. */
export function advance(queue: PackWord[], accepts: Record<string, number>, needed: number): PackWord[] {
  const [current, ...rest] = queue;
  return (accepts[current.sign] ?? 0) >= needed ? rest : [...rest, current];
}
