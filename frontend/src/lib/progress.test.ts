import { describe, expect, it } from "vitest";

import { add, boxes, fresh, known, record, type Learned } from "$lib/progress";

const DAY = 24 * 60 * 60 * 1000;
/** A card: the sign that scores it, its lexicon entry and the word the learner reads. */
const word = (name: string) => ({ word: name, id: name, sign: `sts:${name}-1` });
/** A sign in the store, as `add` wrote it and an answer may since have moved it. */
const stored = (name: string, over: Partial<Learned> = {}): Learned => ({
  box: 0,
  due: 0,
  added: 0,
  word: name,
  id: name,
  ...over,
});
const milk = word("mjölk");

describe("record", () => {
  const picked = { "sts:mjölk-1": stored("mjölk") };

  it("moves an accepted sign up a box and a rejected one back to the first", () => {
    const first = record(picked, milk, true, 0);
    // the first box waits a day, and the time the sign was first met is kept from here on
    expect(first["sts:mjölk-1"]).toEqual(stored("mjölk", { box: 1, due: DAY, seen: 0, first: 0 }));

    // each answer on the day the box before it fell due, since a sign answered early does not move
    const second = record(first, milk, true, DAY);
    expect(second["sts:mjölk-1"]).toEqual(stored("mjölk", { box: 2, due: 4 * DAY, seen: DAY, first: 0 }));

    const third = record(second, milk, true, 4 * DAY);
    expect(third["sts:mjölk-1"]).toEqual(stored("mjölk", { box: 3, due: 11 * DAY, seen: 4 * DAY, first: 0 }));

    const missed = record(third, milk, false, 4 * DAY);
    expect(missed["sts:mjölk-1"]).toEqual(stored("mjölk", { box: 1, due: 5 * DAY, seen: 4 * DAY, first: 0 }));
  });

  it("neither promotes nor reschedules a sign answered before it was due", () => {
    const progress = { "sts:mjölk-1": stored("mjölk", { box: 2, due: 3 * DAY, seen: 0, first: 0 }) };

    // accepted on the first day: the box claims three, which this answer has not tested, so only the
    // time the sign was last seen moves
    expect(record(progress, milk, true, DAY)["sts:mjölk-1"]).toEqual(
      stored("mjölk", { box: 2, due: 3 * DAY, seen: DAY, first: 0 }),
    );

    // a miss is evidence whatever the day: the interval was already too long
    expect(record(progress, milk, false, DAY)["sts:mjölk-1"]).toEqual(
      stored("mjölk", { box: 1, due: 2 * DAY, seen: DAY, first: 0 }),
    );

    // and the same sign, answered on the day it fell due, moves up as before
    expect(record(progress, milk, true, 3 * DAY)["sts:mjölk-1"].box).toBe(3);
  });
});

describe("known", () => {
  // mamma is due and in the first box (weight 1), hej is in the last one and due in three weeks (1/21)
  const progress = {
    "sts:mamma-1": stored("mamma", { box: 1, due: 0, seen: 0 }),
    "sts:hej-1": stored("hej", { box: 4, due: 5 * DAY, seen: 0 }),
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
    const withPicked = { ...progress, "sts:bröd-1": stored("bröd") };

    expect(known(withPicked, 5, () => 0.5)).toEqual([word("mamma"), word("hej")]);
  });
});

describe("add", () => {
  it("puts a picked sign in box 0, where nothing is scheduled until it is taught", () => {
    expect(add({}, [milk], 7)).toEqual({ "sts:mjölk-1": stored("mjölk", { added: 7 }) });
  });

  it("names a sign the search did not name after the sign itself", () => {
    // the server leaves the word out when the sign is named for it, and the store always has one
    expect(add({}, [{ sign: "sts:hund-4711", id: "4711" }], 7)["sts:hund-4711"].word).toBe("hund");
  });

  it("leaves a sign that is already in the store alone", () => {
    // picking a word again must never undo what has been learned of it
    const learned = { "sts:mjölk-1": stored("mjölk", { box: 3, due: 11 * DAY, seen: 0, first: 0 }) };

    expect(add(learned, [milk], 7)).toEqual(learned);
  });
});

describe("fresh", () => {
  it("teaches the picked signs in the order they were picked, and only those", () => {
    const progress = {
      "sts:hej-1": stored("hej", { added: 2 * DAY }),
      "sts:mamma-1": stored("mamma", { added: DAY }),
      "sts:mjölk-1": stored("mjölk", { box: 1, due: DAY, seen: 0, first: 0 }), // already taught
    };

    expect(fresh(progress)).toEqual([word("mamma"), word("hej")]);
  });
});

describe("boxes", () => {
  it("shows every picked and practised sign, the ones not yet taught first", () => {
    const progress = {
      "sts:bröd-1": stored("bröd", { box: 2, due: DAY }),
      "sts:livsmedel-1": stored("äta", { box: 1, due: 0 }), // the word on the card, not the sign's name
      "sts:mamma-1": stored("mamma", { added: DAY }),
    };

    expect(boxes(progress).map((row) => row.word)).toEqual(["mamma", "äta", "bröd"]);
    // box 0 has no due date at all, so a picked sign sorts before everything scheduled
    expect(boxes(progress)[0]).toEqual({ ...stored("mamma", { added: DAY }), sign: "sts:mamma-1" });
  });
});
