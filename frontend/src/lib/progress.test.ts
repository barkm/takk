import { describe, expect, it } from "vitest";

import { boxes, chosenWords, known, record } from "$lib/progress";

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
