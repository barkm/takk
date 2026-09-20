import { describe, expect, it } from "vitest";

import { boxes, chosenWords, practisedToday, record, session } from "$lib/progress";

const word = (name: string) => ({ word: name, id: name, sign: `sts:${name}-1` });
const DAY = 24 * 60 * 60 * 1000;

describe("record", () => {
  it("moves an accepted sign up a box and a rejected one back to the first", () => {
    const first = record({}, "sts:mjölk-1", true, "mjölk", 0);
    expect(first["sts:mjölk-1"]).toEqual({ box: 1, due: DAY, seen: 0, word: "mjölk" }); // the first box waits a day

    const second = record(first, "sts:mjölk-1", true, "mjölk", 0);
    expect(second["sts:mjölk-1"]).toEqual({ box: 2, due: DAY, seen: 0, word: "mjölk" });

    const third = record(second, "sts:mjölk-1", true, "mjölk", 0);
    expect(third["sts:mjölk-1"]).toEqual({ box: 3, due: 3 * DAY, seen: 0, word: "mjölk" });

    const missed = record(third, "sts:mjölk-1", false, "mjölk", 0);
    expect(missed["sts:mjölk-1"]).toEqual({ box: 1, due: DAY, seen: 0, word: "mjölk" });
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
});

describe("chosenWords", () => {
  const packs = [
    { name: "Första tecknen", kind: "pack" as const, words: [{ sign: "sts:mjölk-1", id: "1", word: "mjölk" }] },
    { name: "Mat och dryck", kind: "category" as const, words: [{ sign: "sts:mjölk-1", id: "1" }, { sign: "sts:bröd-2", id: "2" }] },
  ];

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
    { name: "Första tecknen", kind: "pack" as const, words: [{ sign: "sts:livsmedel-1", id: "1", word: "äta" }] },
    { name: "Mat och dryck", kind: "category" as const, words: [{ sign: "sts:livsmedel-1", id: "1" }] },
  ];

  it("shows each practised sign by its pack's word, soonest due first", () => {
    const progress = { "sts:bröd-2": { box: 2, due: DAY }, "sts:livsmedel-1": { box: 1, due: 0 } };

    expect(boxes(progress, packs)).toEqual([
      // no word was stored, so the first pack that names the sign does, not the last one; Mat och
      // dryck holds the same sign as "livsmedel", which is a different word to practise
      { sign: "sts:livsmedel-1", word: "äta", box: 1, due: 0, packs: ["Första tecknen"] },
      // practised but in no pack any more: its own name, which is what the lexicon calls the sign
      { sign: "sts:bröd-2", word: "bröd", box: 2, due: DAY, packs: [] },
    ]);
  });

  it("shows the word that was on the card, not another pack's name for the same sign", () => {
    // sts:spader-00016 is "svart" in Färger and "Oden" in Mytologi: one sign class, two words
    const both = [
      { name: "Färger", kind: "category" as const, words: [{ sign: "sts:spader-1", id: "1", word: "svart" }] },
      { name: "Mytologi", kind: "category" as const, words: [{ sign: "sts:spader-1", id: "2", word: "Oden" }] },
    ];
    const progress = { "sts:spader-1": { box: 2, due: DAY, seen: 0, word: "svart" } };

    // and the packs are the ones that teach "svart", not every pack the sign class appears in
    expect(boxes(progress, both)).toEqual([
      { sign: "sts:spader-1", word: "svart", box: 2, due: DAY, seen: 0, packs: ["Färger"] },
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
