import { describe, expect, it } from "vitest";

import { boxes, chosenWords, record, session } from "$lib/progress";

const word = (name: string) => ({ word: name, id: name, sign: `sts:${name}-1` });
const DAY = 24 * 60 * 60 * 1000;

describe("record", () => {
  it("moves an accepted sign up a box and a rejected one back to the first", () => {
    const first = record({}, "sts:mjölk-1", true, 0);
    expect(first["sts:mjölk-1"]).toEqual({ box: 1, due: 0 }); // the first box is due in the same session

    const second = record(first, "sts:mjölk-1", true, 0);
    expect(second["sts:mjölk-1"]).toEqual({ box: 2, due: DAY });

    expect(record(second, "sts:mjölk-1", false, 0)["sts:mjölk-1"]).toEqual({ box: 1, due: 0 });
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
      { sign: "sts:livsmedel-1", word: "äta", box: 1, due: 0, packs: ["Första tecknen", "Mat och dryck"] },
      // practised but in no pack any more: its own name, which is what the lexicon calls the sign
      { sign: "sts:bröd-2", word: "bröd", box: 2, due: DAY, packs: [] },
    ]);
  });
});
