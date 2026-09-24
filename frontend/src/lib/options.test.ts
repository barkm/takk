import { describe, expect, test } from "vitest";

import { pass } from "./options";
import type { SignWord } from "./api";

const words: SignWord[] = ["hund", "katt", "bok"].map((word) => ({ sign: `sts:${word}-1`, id: word, word }));

describe("pass", () => {
  test("signs every word once when nothing is repeated", () => {
    expect(pass(words, 1).map((each) => each.word)).toEqual(["hund", "katt", "bok"]);
  });

  test("brings a word back with the others in between, not straight after itself", () => {
    expect(pass(words, 3).map((each) => each.word)).toEqual(
      ["hund", "katt", "bok", "hund", "katt", "bok", "hund", "katt", "bok"],
    );
  });

  test("is empty when there is nothing to practise", () => {
    expect(pass([], 3)).toEqual([]);
  });
});
