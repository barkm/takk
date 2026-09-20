import { describe, expect, it } from "vitest";

import { type PackWord } from "$lib/api";
import { advance, boxes, chosenWords, loadSetting, practisedToday, record, saveSetting, session } from "$lib/progress";

const word = (name: string) => ({
  word: name,
  id: name,
  sign: `sts:${name}-1`,
});
const DAY = 24 * 60 * 60 * 1000;

describe("record", () => {
  it("moves an accepted sign up a box and a rejected one back to the first", () => {
    const first = record({}, "sts:mjölk-1", true, "mjölk", 0);
    // the first box waits a day, and the time the sign was first met is kept from here on
    expect(first["sts:mjölk-1"]).toEqual({
      box: 1,
      due: DAY,
      seen: 0,
      first: 0,
      word: "mjölk",
    });

    const second = record(first, "sts:mjölk-1", true, "mjölk", 0);
    expect(second["sts:mjölk-1"]).toEqual({
      box: 2,
      due: DAY,
      seen: 0,
      first: 0,
      word: "mjölk",
    });

    const third = record(second, "sts:mjölk-1", true, "mjölk", 0);
    expect(third["sts:mjölk-1"]).toEqual({
      box: 3,
      due: 3 * DAY,
      seen: 0,
      first: 0,
      word: "mjölk",
    });

    const missed = record(third, "sts:mjölk-1", false, "mjölk", 0);
    expect(missed["sts:mjölk-1"]).toEqual({
      box: 1,
      due: DAY,
      seen: 0,
      first: 0,
      word: "mjölk",
    });
  });
});

describe("session", () => {
  const words = [word("mamma"), word("pappa"), word("hej")];

  it("takes the due signs first and fills up with signs never practised", () => {
    const progress = {
      "sts:mamma-1": { box: 2, due: 5 * DAY }, // not due yet
      "sts:pappa-1": { box: 1, due: 0 }, // due
    };

    expect(session(words, progress, 2, DAY)).toEqual([word("pappa"), word("hej")]);
  });

  it("is the new signs alone when nothing was practised, at most `size` of them", () => {
    expect(session(words, {}, 2)).toEqual([word("mamma"), word("pappa")]);
  });

  it("leaves no room for a new sign when the pass is full of repetitions", () => {
    const progress = {
      "sts:mamma-1": { box: 1, due: DAY }, // due later today than pappa
      "sts:pappa-1": { box: 1, due: 0 },
    };

    // soonest due first, and "hej", never practised, does not fit in a pass of two
    expect(session(words, progress, 2, 2 * DAY)).toEqual([word("pappa"), word("mamma")]);
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

  it("keeps a word in the pass until it has been accepted enough times", () => {
    // accepted once of two, so it goes back to the end rather than leaving the pass
    expect(advance(queue, { "sts:mamma-1": 1 }, 2)).toEqual([word("pappa"), word("mamma")]);
    // a missed word has no accepts at all, and comes back the same way
    expect(advance(queue, {}, 2)).toEqual([word("pappa"), word("mamma")]);
    expect(advance(queue, { "sts:mamma-1": 2 }, 2)).toEqual([word("pappa")]);
  });

  it("empties on the last word of the pass", () => {
    expect(advance([word("hej")], { "sts:hej-1": 1 }, 1)).toEqual([]);
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
    queue = advance(queue, accepts, needed);
    expect(queue).toEqual([word("pappa"), word("mamma")]);

    accepts = { ...accepts, "sts:pappa-1": 0 }; // missed: back to the first box, and back in the queue
    progress = record(progress, "sts:pappa-1", false, "pappa", 0);
    queue = advance(queue, accepts, needed);
    expect(queue).toEqual([word("mamma"), word("pappa")]);

    accepts = { ...accepts, "sts:mamma-1": 2 }; // accepted twice: finished, and out of the queue
    progress = record(progress, "sts:mamma-1", true, "mamma", 0);
    expect(progress).toEqual({
      "sts:pappa-1": { box: 1, due: DAY, seen: 0, first: 0, word: "pappa" },
      "sts:mamma-1": { box: 1, due: DAY, seen: 0, first: 0, word: "mamma" },
    });
    expect(advance(queue, accepts, needed)).toEqual([word("pappa")]);
  });

  it("leaves a word the pass missed in the first box, however well it ends", () => {
    // A miss puts the word in the first box, and finishing the pass must not carry it back out:
    // otherwise missing a word would promote it further than signing it right the first time.
    const known = { "sts:pappa-1": { box: 4, due: 0, seen: 0, first: 0, word: "pappa" } };
    const missed = record(known, "sts:pappa-1", false, "pappa", 0);
    expect(missed["sts:pappa-1"].box).toBe(1);

    // the page passes `!missed.includes(sign)`, so the word finishes where the miss left it
    expect(record(missed, "sts:pappa-1", false, "pappa", 0)["sts:pappa-1"].box).toBe(1);
    // and a word the pass never missed climbs the one box it earned
    expect(record(known, "sts:pappa-1", true, "pappa", 0)["sts:pappa-1"].box).toBe(5);
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
