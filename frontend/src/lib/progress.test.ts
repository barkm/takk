import { describe, expect, it } from "vitest";

import { type PackWord } from "$lib/api";
import { advance, boxes, chosenWords, known, loadSetting, practisedToday, record, saveSetting, session, turn } from "$lib/progress";  // prettier-ignore

const word = (name: string) => ({
  word: name,
  id: name,
  sign: `sts:${name}-1`,
});
const DAY = 24 * 60 * 60 * 1000;
const milk = { sign: "sts:mjölk-1", id: "1", word: "mjölk" }; // a card: the sign, its lexicon entry and its word

describe("record", () => {
  it("moves an accepted sign up a box and a rejected one back to the first", () => {
    const first = record({}, milk, true, 0);
    // the first box waits a day, and the time the sign was first met is kept from here on
    expect(first["sts:mjölk-1"]).toEqual({
      box: 1,
      due: DAY,
      seen: 0,
      first: 0,
      word: "mjölk",
      id: "1",
    });

    // each answer on the day the box before it fell due, since a sign answered early does not move
    const second = record(first, milk, true, DAY);
    expect(second["sts:mjölk-1"]).toEqual({
      box: 2,
      due: 4 * DAY,
      seen: DAY,
      first: 0,
      word: "mjölk",
      id: "1",
    });

    const third = record(second, milk, true, 4 * DAY);
    expect(third["sts:mjölk-1"]).toEqual({
      box: 3,
      due: 11 * DAY,
      seen: 4 * DAY,
      first: 0,
      word: "mjölk",
      id: "1",
    });

    const missed = record(third, milk, false, 4 * DAY);
    expect(missed["sts:mjölk-1"]).toEqual({
      box: 1,
      due: 5 * DAY,
      seen: 4 * DAY,
      first: 0,
      word: "mjölk",
      id: "1",
    });
  });

  it("neither promotes nor reschedules a sign answered before it was due", () => {
    const progress = { "sts:mjölk-1": { box: 2, due: 3 * DAY, seen: 0, first: 0, word: "mjölk" } };

    // accepted on the first day: the box claims three, which this answer has not tested, so only the
    // time the sign was last seen moves
    expect(record(progress, milk, true, DAY)["sts:mjölk-1"]).toEqual({
      box: 2,
      due: 3 * DAY,
      seen: DAY,
      first: 0,
      word: "mjölk",
      id: "1",
    });

    // a miss is evidence whatever the day: the interval was already too long
    expect(record(progress, milk, false, DAY)["sts:mjölk-1"]).toEqual({
      box: 1,
      due: 2 * DAY,
      seen: DAY,
      first: 0,
      word: "mjölk",
      id: "1",
    });

    // and the same sign, answered on the day it fell due, moves up as before
    expect(record(progress, milk, true, 3 * DAY)["sts:mjölk-1"].box).toBe(3);
  });
});

describe("known", () => {
  const packs = [{ name: "Första tecknen", kind: "pack" as const, words: [word("mamma"), word("hej")] }];
  // mamma is due and in the first box (weight 1), hej is in the last one and due in three weeks (1/21)
  const progress = {
    "sts:mamma-1": { box: 1, due: 0, seen: 0, word: "mamma", id: "mamma" },
    "sts:hej-1": { box: 4, due: 5 * DAY, seen: 0, word: "hej", id: "hej" },
  };

  it("draws the weak words far more often than the strong ones", () => {
    // the weights add up to 1 + 1/21, so all but the last 4.5% of the range falls on the first box
    expect(known(packs, progress, 1, () => 0.5)).toEqual([word("mamma")]);
    expect(known(packs, progress, 1, () => 0.99)).toEqual([word("hej")]);
  });

  it("draws without replacement, and stops when the practised words run out", () => {
    expect(known(packs, progress, 5, () => 0.5)).toEqual([word("mamma"), word("hej")]);
    expect(known(packs, {}, 5)).toEqual([]);
  });

  it("practises a sign that is in no pack any more, by the card the box kept", () => {
    // the word and the lexicon entry come from the store, so a pack the crawl dropped changes nothing
    expect(known([], progress, 1, () => 0)).toEqual([word("mamma")]);
    // a box written before the entry was stored still practises, with no entry to describe its form
    const old = { "sts:hej-1": { box: 1, due: 0, word: "hej" } };
    expect(known([], old, 1, () => 0)).toEqual([{ sign: "sts:hej-1", id: "", word: "hej" }]);
  });
});

describe("session", () => {
  const packs = [{ name: "Första tecknen", kind: "pack" as const, words: [word("mamma"), word("pappa"), word("hej")] }];  // prettier-ignore
  const chosen = ["Första tecknen"];
  const learned = (sign: string, due: number, box = 1) => ({ [sign]: { box, due, word: sign.slice(4, -2), id: sign.slice(4, -2) } });  // prettier-ignore

  it("takes the due signs first and fills up with signs never practised", () => {
    const progress = {
      ...learned("sts:mamma-1", 5 * DAY, 2), // not due yet
      ...learned("sts:pappa-1", 0), // due
    };

    expect(session(packs, chosen, progress, 2, DAY)).toEqual([word("pappa"), word("hej")]);
  });

  it("is the new signs alone when nothing was practised, at most `size` of them", () => {
    expect(session(packs, chosen, {}, 2)).toEqual([word("mamma"), word("pappa")]);
  });

  it("leaves no room for a new sign when the pass is full of repetitions", () => {
    const progress = {
      ...learned("sts:mamma-1", DAY), // due later today than pappa
      ...learned("sts:pappa-1", 0),
    };

    // soonest due first, and "hej", never practised, does not fit in a pass of two
    expect(session(packs, chosen, progress, 2, 2 * DAY)).toEqual([word("pappa"), word("mamma")]);
  });

  it("repeats a due sign that no chosen pack teaches, and teaches new signs from the chosen ones", () => {
    // the packs say where new words come from, nothing more: a word already learned is still repeated
    const progress = learned("sts:röd-1", 0);

    expect(session(packs, [], progress, 5, DAY)).toEqual([word("röd")]);
    expect(session(packs, chosen, progress, 5, DAY)).toEqual([word("röd"), word("mamma"), word("pappa"), word("hej")]);  // prettier-ignore
  });
});

describe("chosenWords", () => {
  const packs = [
    {
      name: "Första tecknen",
      kind: "pack" as const,
      words: [{ sign: "sts:mjölk-1", id: "1", word: "mjölk" }],
    },
    {
      name: "Mat och dryck",
      kind: "category" as const,
      words: [
        { sign: "sts:mjölk-1", id: "1" },
        { sign: "sts:bröd-2", id: "2" },
      ],
    },
  ];

  it("takes a word from each chosen pack in turn, so a long pack cannot hide a short one", () => {
    const long = {
      name: "Djur",
      kind: "category" as const,
      words: [word("hund"), word("katt"), word("orm")],
    };
    const short = {
      name: "Färger",
      kind: "category" as const,
      words: [word("röd")],
    };

    expect(chosenWords([long, short], ["Djur", "Färger"]).map((each) => each.word)).toEqual([
      "hund",
      "röd", // the short pack's only word comes second, not fourth
      "katt",
      "orm",
    ]);
  });

  it("takes the chosen packs only, and a sign in two of them once", () => {
    expect(chosenWords(packs, ["Första tecknen", "Mat och dryck"])).toEqual([
      { sign: "sts:mjölk-1", id: "1", word: "mjölk" }, // the word of the pack that named it wins
      { sign: "sts:bröd-2", id: "2" },
    ]);
    expect(chosenWords(packs, [])).toEqual([]);
  });
});

describe("boxes", () => {
  const packs = [
    {
      name: "Första tecknen",
      kind: "pack" as const,
      words: [{ sign: "sts:livsmedel-1", id: "1", word: "äta" }],
    },
    {
      name: "Mat och dryck",
      kind: "category" as const,
      words: [{ sign: "sts:livsmedel-1", id: "1" }],
    },
  ];

  it("shows each practised sign by its pack's word, soonest due first", () => {
    const progress = {
      "sts:bröd-2": { box: 2, due: DAY },
      "sts:livsmedel-1": { box: 1, due: 0 },
    };

    expect(boxes(progress, packs)).toEqual([
      // no word was stored, so the first pack that names the sign does, not the last one; Mat och
      // dryck holds the same sign as "livsmedel", which is a different word to practise
      {
        sign: "sts:livsmedel-1",
        word: "äta",
        box: 1,
        due: 0,
        packs: ["Första tecknen"],
      },
      // practised but in no pack any more: its own name, which is what the lexicon calls the sign
      { sign: "sts:bröd-2", word: "bröd", box: 2, due: DAY, packs: [] },
    ]);
  });

  it("shows the word that was on the card, not another pack's name for the same sign", () => {
    // sts:spader-00016 is "svart" in Färger and "Oden" in Mytologi: one sign class, two words
    const both = [
      {
        name: "Färger",
        kind: "category" as const,
        words: [{ sign: "sts:spader-1", id: "1", word: "svart" }],
      },
      {
        name: "Mytologi",
        kind: "category" as const,
        words: [{ sign: "sts:spader-1", id: "2", word: "Oden" }],
      },
    ];
    const progress = {
      "sts:spader-1": { box: 2, due: DAY, seen: 0, word: "svart" },
    };

    // and the packs are the ones that teach "svart", not every pack the sign class appears in
    expect(boxes(progress, both)).toEqual([
      {
        sign: "sts:spader-1",
        word: "svart",
        box: 2,
        due: DAY,
        seen: 0,
        packs: ["Färger"],
      },
    ]);
  });
});

describe("practisedToday", () => {
  it("counts the signs practised since midnight, whenever in the day it is asked", () => {
    const noon = new Date("2026-09-20T12:00:00").getTime();
    const progress = {
      "sts:mjölk-1": { box: 2, due: noon + DAY, seen: noon - 60_000 },
      "sts:bröd-2": { box: 1, due: noon, seen: noon - 20 * 60 * 60 * 1000 }, // practised yesterday
      "sts:ost-3": { box: 3, due: noon }, // a box written before the app kept the time
    };

    expect(practisedToday(progress, noon)).toBe(1);
  });
});

describe("advance", () => {
  const queue = [word("mamma"), word("pappa")];
  const head = [word("mamma")];

  it("keeps a word in the pass until it has been accepted enough times", () => {
    // accepted once of two, so it goes back to the end rather than leaving the pass
    expect(advance(queue, head, { "sts:mamma-1": 1 }, 2)).toEqual([word("pappa"), word("mamma")]);
    // a missed word has no accepts at all, and comes back the same way
    expect(advance(queue, head, {}, 2)).toEqual([word("pappa"), word("mamma")]);
    expect(advance(queue, head, { "sts:mamma-1": 2 }, 2)).toEqual([word("pappa")]);
  });

  it("takes every word of a sentence out of the queue, and puts back the unfinished ones", () => {
    const three = [word("mamma"), word("pappa"), word("hej")];
    const sentence = [word("mamma"), word("hej")]; // a turn need not be the front of the queue
    expect(advance(three, sentence, { "sts:mamma-1": 2, "sts:hej-1": 1 }, 2)).toEqual([word("pappa"), word("hej")]);
  });

  it("empties on the last word of the pass", () => {
    expect(advance([word("hej")], [word("hej")], { "sts:hej-1": 1 }, 1)).toEqual([]);
  });
});

describe("turn", () => {
  const queue = [word("mamma"), word("pappa"), word("hej"), word("tack")];
  const known = (box: number) => ({ box, due: 0 });

  it("is the head alone while the head is still in the first box", () => {
    // a new word, or one missed today: it is being taught or has just gone wrong, so not in a sentence
    expect(turn(queue, {})).toEqual([word("mamma")]);
    expect(turn(queue, { "sts:mamma-1": known(1), "sts:pappa-1": known(3) })).toEqual([word("mamma")]);
  });

  it("offers every word that may be combined once the head is one of them", () => {
    const progress = { "sts:mamma-1": known(2), "sts:pappa-1": known(1), "sts:hej-1": known(4), "sts:tack-1": known(2) };  // prettier-ignore
    // pappa is in the first box, so it is skipped over rather than buried in the sentence; which of
    // the other three the sentence uses is the model's choice, and only the head is required
    expect(turn(queue, progress)).toEqual([word("mamma"), word("hej"), word("tack")]);
  });

  it("is empty when the pass is", () => {
    expect(turn([], {})).toEqual([]);
  });
});

describe("a pass, turn by turn", () => {
  // What the page does with these: a word is stored only once its fate is settled, so two accepts
  // move it up a box, and one accept and a miss leave it in the first box, still in the pass.
  const needed = 2;

  it("moves a word up a box when it has been accepted enough times, and not before", () => {
    let queue: PackWord[] = [word("mamma"), word("pappa")];
    let accepts: Record<string, number> = {};
    let progress = {};

    accepts = { ...accepts, "sts:mamma-1": 1 }; // accepted once: nothing stored yet
    expect(progress).toEqual({});
    queue = advance(queue, [queue[0]], accepts, needed);
    expect(queue).toEqual([word("pappa"), word("mamma")]);

    accepts = { ...accepts, "sts:pappa-1": 0 }; // missed: back to the first box, and back in the queue
    progress = record(progress, word("pappa"), false, 0);
    queue = advance(queue, [queue[0]], accepts, needed);
    expect(queue).toEqual([word("mamma"), word("pappa")]);

    accepts = { ...accepts, "sts:mamma-1": 2 }; // accepted twice: finished, and out of the queue
    progress = record(progress, word("mamma"), true, 0);
    expect(progress).toEqual({
      "sts:pappa-1": { box: 1, due: DAY, seen: 0, first: 0, word: "pappa", id: "pappa" },
      "sts:mamma-1": { box: 1, due: DAY, seen: 0, first: 0, word: "mamma", id: "mamma" },
    });
    expect(advance(queue, [queue[0]], accepts, needed)).toEqual([word("pappa")]);
  });

  it("leaves a word the pass missed in the first box, however well it ends", () => {
    // A miss puts the word in the first box, and finishing the pass must not carry it back out:
    // otherwise missing a word would promote it further than signing it right the first time.
    const known = { "sts:pappa-1": { box: 3, due: 0, seen: 0, first: 0, word: "pappa" } };
    const missed = record(known, word("pappa"), false, 0);
    expect(missed["sts:pappa-1"].box).toBe(1);

    // the page passes `!missed.includes(sign)`, so the word finishes where the miss left it
    expect(record(missed, word("pappa"), false, 0)["sts:pappa-1"].box).toBe(1);
    // and a word the pass never missed climbs the one box it earned
    expect(record(known, word("pappa"), true, 0)["sts:pappa-1"].box).toBe(4);
  });
});

describe("loadSetting", () => {
  it("keeps a chosen number and refuses one a pass cannot be run with", () => {
    const stored = new Map<string, string>(); // the tests run without a browser
    Object.assign(globalThis, {
      localStorage: {
        getItem: (key: string) => stored.get(key) ?? null,
        setItem: (key: string, value: string) => stored.set(key, value),
      },
    });

    expect(loadSetting("size")).toBe(5); // never chosen
    expect(loadSetting("accepts")).toBe(2);

    saveSetting("size", 12);
    expect(loadSetting("size")).toBe(12);

    saveSetting("accepts", 0); // a word accepted no times would never finish
    expect(loadSetting("accepts")).toBe(1);
  });
});
