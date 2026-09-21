// What the learner has picked and practised, kept in the browser (step 8 of ROADMAP-takk.md): no
// accounts and no state on the server for as long as that holds. A sign enters the store in box 0
// when it is ticked in Tecken, moves into the first box when Nya ord teaches it, up a box when a due
// sign is signed right, and back to the first box the moment it is missed. The story is where a
// learned sign is repeated (see steps 9 and 13 of ROADMAP-takk.md).
import { word as signWord, type SignWord } from "$lib/api";

export const KEY = "takk.progress";
const DAY = 24 * 60 * 60 * 1000;
/** Days until a sign in each box is due again, and how many boxes there are (user, 2026-09-21).
 * Changing this array is the whole schedule, and `Math.min` in `record` keeps a word in the last box:
 * a box beyond the array, left in a store by a longer schedule, is clamped by the next answer. */
export const DAYS = [1, 3, 7, 21];

/** A sign in the store: its box, when it is due again, when it was picked, and the word and lexicon
 * entry of the card it was picked as, in milliseconds since the epoch.
 *
 * The word is kept because a box belongs to a sign class and a class has many words:
 * `sts:spader-00016` is "svart", "lakrits" and "spader" at once, and what was practised is the one
 * that was on the card. The entry is kept for the same reason and so that nothing has to be looked up
 * to practise the word again: the lexicon is the server's and the boxes are the browser's, and the
 * entry is what a card's description of the sign form is fetched by.
 *
 * Box 0 is a sign picked in Tecken and not practised yet, which is where every sign enters the store
 * (step 9 of ROADMAP-takk.md); it has no `due`, since nothing about it is scheduled until it is
 * answered, and no `seen` or `first`, which an answer writes. `first` is when the sign was met, the
 * only way to tell a new word from a missed old one once both sit in the first box. */
export type Learned = { box: number; due: number; added: number; word: string; id: string; seen?: number; first?: number };

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
export function record(progress: Progress, card: SignWord, correct: boolean, now = Date.now()): Progress {
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
  // the word and the entry are the card's, which is the word practised and not the name of the sign
  const learned: Learned = { ...previous, box, due, seen: now, first: previous?.first ?? now };
  return { ...progress, [card.sign]: learned };
}

/** The progress after the learner picked `words` in Tecken. A sign already in the store keeps its
 * box, so picking a word again never undoes what has been learned of it; a new one lands in box 0,
 * which is "valt men inte övat" and what Nya ord teaches from. */
export function add(progress: Progress, words: SignWord[], now = Date.now()): Progress {
  const picked = { ...progress };
  for (const each of words)
    // the word the sign is offered under is resolved here, so nothing downstream has to name a sign
    if (!picked[each.sign])
      picked[each.sign] = { box: 0, due: 0, added: now, word: each.word ?? signWord(each.sign), id: each.id };
  return picked;
}

/** The signs picked but never practised, the longest waiting first, which is the order Nya ord
 * teaches them in. */
export function fresh(progress: Progress): SignWord[] {
  return boxes(progress)
    .filter((row) => row.box === 0)
    .sort((a, b) => a.added - b.added)
    .map(card);
}

/** A sign in the store, as the progress page shows it: soonest due first, so the ones picked and not
 * yet practised come first of all. */
export type Row = Learned & { sign: string };

export function boxes(progress: Progress): Row[] {
  const rows = Object.entries(progress).map(([sign, learned]) => ({ ...learned, sign }));
  return rows.sort((a, b) => a.due - b.due || a.word.localeCompare(b.word, "sv"));
}

/** The card a box holds: its word and its lexicon entry, both stored with it, so nothing is looked
 * up to practise a word again. */
const card = (row: Row): SignWord => ({
  sign: row.sign,
  id: row.id ?? "",
  word: row.word,
});

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
export function known(progress: Progress, size: number, random = Math.random): SignWord[] {
  const pool = boxes(progress)
    .filter((row) => row.box > 0) // a sign picked but never practised is taught first, not repeated
    .map((row) => ({
      word: card(row),
      weight: 1 / DAYS[Math.min(row.box, DAYS.length) - 1],
    }));
  const picked: SignWord[] = [];
  while (picked.length < size && pool.length) {
    let point = random() * pool.reduce((sum, each) => sum + each.weight, 0);
    const at = pool.findIndex((each) => (point -= each.weight) < 0);
    picked.push(pool.splice(at < 0 ? pool.length - 1 : at, 1)[0].word); // `at < 0` only by rounding
  }
  return picked;
}
