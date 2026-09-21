import { describe, expect, it } from "vitest";

import { pieces } from "$lib/text";

describe("pieces", () => {
  const marked = (text: string, words: string[]) =>
    pieces(text, words)
      .filter((piece) => piece.key)
      .map((piece) => piece.text);

  it("marks a word whatever letters it is written with", () => {
    // the bug this exists for: JavaScript's \b is ASCII-only, so \bblå\b matched nothing
    expect(marked("Jag har en blå bil.", ["blå"])).toEqual(["blå"]);
    expect(marked("Mamma vill sova.", ["mamma", "sova"])).toEqual(["Mamma", "sova"]);
    expect(marked("Vi ska äta platta slag idag.", ["äta", "platta slag"])).toEqual(["äta", "platta slag"]);
  });

  it("marks whole words only", () => {
    expect(marked("Vi plockar blåbär.", ["blå"])).toEqual([]);
    expect(marked("Jag drack mjölken.", ["mjölk"])).toEqual([]);
  });

  it("keeps the spoken text around them", () => {
    expect(pieces("En blå bil", ["blå"])).toEqual([
      { text: "En ", key: false },
      { text: "blå", key: true },
      { text: " bil", key: false },
    ]);
  });
});
