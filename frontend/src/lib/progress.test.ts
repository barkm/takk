import { describe, expect, it } from "vitest";

import { add, boxes, fresh, known, record } from "$lib/progress";

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
    const progress = {
      "sts:mjölk-1": { box: 2, due: 3 * DAY, seen: 0, first: 0, word: "mjölk" },
    };

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
  // mamma is due and in the first box (weight 1), hej is in the last one and due in three weeks (1/21)
  const progress = {
    "sts:mamma-1": { box: 1, due: 0, seen: 0, word: "mamma", id: "mamma" },
    "sts:hej-1": { box: 4, due: 5 * DAY, seen: 0, word: "hej", id: "hej" },
  };

  it("draws the weak words far more often than the strong ones", () => {
    // the weights add up to 1 + 1/21, so all but the last 4.5% of the range falls on the first box
    expect(known(progress, 1, () => 0.5)).toEqual([word("mamma")]);
    expect(known(progress, 1, () => 0.99)).toEqual([word("hej")]);
  });

  it("draws without replacement, and stops when the practised words run out", () => {
    expect(known(progress, 5, () => 0.5)).toEqual([word("mamma"), word("hej")]);
    expect(known({}, 5)).toEqual([]);
  });

  it("never repeats a sign that was only picked, since it has not been taught yet", () => {
    const picked = {
      ...progress,
      "sts:bröd-2": { box: 0, due: 0, added: 0, word: "bröd", id: "2" },
    };

    expect(known(picked, 5, () => 0.5)).toEqual([word("mamma"), word("hej")]);
  });

  it("practises by the card the box kept, whatever the lexicon says now", () => {
    // a box written before the entry was stored still practises, with no entry to describe its form
    const old = { "sts:hej-1": { box: 1, due: 0, word: "hej" } };

    expect(known(old, 1, () => 0)).toEqual([
      { sign: "sts:hej-1", id: "", word: "hej" },
    ]);
  });
});

describe("add", () => {
  const milkWord = { sign: "sts:mjölk-1", id: "1", word: "mjölk" };

  it("puts a picked sign in box 0, where nothing is scheduled until it is taught", () => {
    expect(add({}, [milkWord], 7)).toEqual({
      "sts:mjölk-1": { box: 0, due: 0, added: 7, word: "mjölk", id: "1" },
    });
  });

  it("leaves a sign that is already in the store alone", () => {
    // picking a word again must never undo what has been learned of it
    const learned = {
      "sts:mjölk-1": {
        box: 3,
        due: 11 * DAY,
        seen: 0,
        first: 0,
        word: "mjölk",
        id: "1",
      },
    };

    expect(add(learned, [milkWord], 7)).toEqual(learned);
  });
});

describe("fresh", () => {
  it("teaches the picked signs in the order they were picked, and only those", () => {
    const progress = {
      "sts:hej-1": { box: 0, due: 0, added: 2 * DAY, word: "hej", id: "hej" },
      "sts:mamma-1": { box: 0, due: 0, added: DAY, word: "mamma", id: "mamma" },
      "sts:mjölk-1": {
        box: 1,
        due: DAY,
        seen: 0,
        first: 0,
        word: "mjölk",
        id: "1",
      }, // already taught
    };

    expect(fresh(progress)).toEqual([word("mamma"), word("hej")]);
  });
});

describe("boxes", () => {
  it("shows every picked and practised sign, the ones not yet taught first", () => {
    const progress = {
      "sts:bröd-2": { box: 2, due: DAY },
      "sts:livsmedel-1": { box: 1, due: 0, word: "äta", id: "1" },
      "sts:mamma-1": { box: 0, due: 0, added: DAY, word: "mamma", id: "mamma" },
    };

    expect(boxes(progress)).toEqual([
      // box 0 has no due date at all, so a picked sign sorts before everything scheduled
      {
        sign: "sts:mamma-1",
        word: "mamma",
        box: 0,
        due: 0,
        added: DAY,
        id: "mamma",
      },
      // the word on the card, which is not the name of the sign that scores it
      { sign: "sts:livsmedel-1", word: "äta", box: 1, due: 0, id: "1" },
      // a box written before the word was stored falls back to what the lexicon calls the sign
      { sign: "sts:bröd-2", word: "bröd", box: 2, due: DAY },
    ]);
  });
});
