// What the learner has practised, kept in the browser (step 8 of ROADMAP-takk.md): no accounts and
// no state on the server for as long as that holds. One Leitner box per sign, saying how long until
// it is due: a word moves up a box when a due sign is signed right, and back to the first box the
// moment it is missed. New words enter the first box on the "Nya ord" page, and the story is where
// they are repeated (see step 9 and step 12 of ROADMAP-takk.md).
import { word as signWord, type Pack, type PackWord } from "$lib/api";

const KEY = "takk.progress";
const CHOSEN = "takk.packs";
const DAY = 24 * 60 * 60 * 1000;
/** Days until a sign in each box is due again, and how many boxes there are (user, 2026-09-21).
 * Changing this array is the whole schedule, and `Math.min` in `record` keeps a word in the last box:
 * a box beyond the array, left in a store by a longer schedule, is clamped by the next answer. */
export const DAYS = [1, 3, 7, 21];

/** A sign's box (1 and up), when it is due again and when it was last practised, in milliseconds
 * since the epoch, and the word the learner was shown. The word is kept because a box belongs to a
 * sign class and a class has many words: `sts:spader-00016` is "svart" in Färger and "Oden" in
 * Mytologi, and what was practised is the one that was on the card. `id` is that card's lexicon entry,
 * kept for the same reason and so that nothing has to be looked up in the packs to practise the word
 * again: the packs come from the crawl and the boxes from `localStorage`, so an entry dropped from
 * every pack would otherwise lose the form description its card shows. `first` is when the sign was
 * met for the first time, which is the only way to tell a new word from a missed old one: both sit in
 * the first box. Boxes written before these fields existed count as neither practised nor met today,
 * fall back to a pack's word, and are given an entry the next time the word is practised. */
export type Learned = { box: number; due: number; seen?: number; first?: number; word?: string; id?: string };

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

/** The progress after `card` was signed right or missed. A word met for the first time and signed
 * right lands in the first box, which is what "Nya ord" records; every later answer comes from a
 * story. */
export function record(progress: Progress, card: PackWord, correct: boolean, now = Date.now()): Progress {
  const previous = progress[card.sign];
  // A sign answered before it was due keeps its box and its date (user, 2026-09-21). A box is a claim
  // about an interval — the third means "still remembered after seven days" — and an answer on the
  // second day has not tested that, so promoting would push the next repetition out to an interval
  // that was never met, and a word could climb out of the boxes by being drilled in one sitting. A
  // miss is evidence whatever the day, since forgetting a sign that was not due means its interval
  // was already too long, so a miss always falls back to the first box.
  const early = correct && previous && previous.due > now;
  const box = early ? previous.box : correct ? Math.min((previous?.box ?? 0) + 1, DAYS.length) : 1;
  const due = early ? previous.due : now + DAYS[box - 1] * DAY;
  const learned: Learned = { box, due, seen: now, first: previous?.first ?? now };
  if (card.word) learned.word = card.word; // the word on the card, not the name of the sign that scores it
  if (card.id) learned.id = card.id;
  return { ...progress, [card.sign]: learned };
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
 * The packs are read in turn rather than one after the other because "Nya ord" takes its words from
 * the front: in order, Djur's 196 words would come before Mat och dryck's first one, which at
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

/** The card a box holds: its word and its lexicon entry, which are stored with it, so a pass needs the
 * packs only to name a box written before the word was kept. */
const card = (row: Row): PackWord => ({ sign: row.sign, id: row.id ?? "", word: row.word });

/** The signs a story is written over: `size` of the words already practised, due or not, drawn without
 * replacement and each weighted by `1 / DAYS[box - 1]`, so a word in the first box is twenty-one times
 * as likely as one in the last (user, 2026-09-21). That weight is the rate the ordinary schedule would
 * meet the word at, so a story is the same schedule run early and spends its words on the
 * weakest ones. Taking the longest unseen instead would do the opposite: the last box is due in three
 * weeks, so its words are always the ones longest unseen.
 *
 * Nothing new is taught here, the pool being what the learner has practised, and `record` leaves a word
 * that was not due where it is, so a story can repair the boxes but never inflate them. Covering
 * every word is the ordinary schedule's job, whatever this samples. `random` is a parameter so that a
 * test can be deterministic. */
export function known(packs: Pack[], progress: Progress, size: number, random = Math.random): PackWord[] {
  const pool = boxes(progress, packs).map((row) => ({
    word: card(row),
    weight: 1 / DAYS[Math.min(row.box, DAYS.length) - 1],
  }));
  const picked: PackWord[] = [];
  while (picked.length < size && pool.length) {
    let point = random() * pool.reduce((sum, each) => sum + each.weight, 0);
    const at = pool.findIndex((each) => (point -= each.weight) < 0);
    picked.push(pool.splice(at < 0 ? pool.length - 1 : at, 1)[0].word); // `at < 0` only by rounding
  }
  return picked;
}
