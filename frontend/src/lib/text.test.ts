import { describe, expect, it } from "vitest";

import { pieces } from "$lib/text";

describe("pieces", () => {
  const marked = (text: string, forms: string[]) =>
    pieces(text, forms)
      .filter((piece) => piece.at !== null)
      .map((piece) => piece.text);

  it("marks a form whatever letters it is written with", () => {
    // the bug this exists for: JavaScript's \b is ASCII-only, so \bblå\b matched nothing
    expect(marked("Jag har en blå bil.", ["blå"])).toEqual(["blå"]);
    expect(marked("Mamma vill sova.", ["Mamma", "sova"])).toEqual(["Mamma", "sova"]);
    expect(marked("Vi ska äta platta slag idag.", ["äta", "platta slag"])).toEqual(["äta", "platta slag"]);
  });

  it("marks the form the story wrote, inflected as it stands", () => {
    expect(marked("Jag drack mjölken.", ["mjölken"])).toEqual(["mjölken"]);
    expect(marked("Barnet sov gott.", ["sov"])).toEqual(["sov"]);
  });

  it("marks whole words only", () => {
    expect(marked("Vi plockar blåbär.", ["blå"])).toEqual([]);
    expect(marked("Jag drack mjölken.", ["mjölk"])).toEqual([]);
  });

  it("marks each occurrence as the sign it is", () => {
    // a form said twice is signed twice, so the second is found after the first, not on top of it
    expect(pieces("Mer mjölk, mer mjölk!", ["mjölk", "mjölk"])).toEqual([
      { text: "Mer ", at: null },
      { text: "mjölk", at: 0 },
      { text: ", mer ", at: null },
      { text: "mjölk", at: 1 },
      { text: "!", at: null },
    ]);
  });

  it("leaves an occurrence the story did not sign unmarked", () => {
    // said twice, signed once: the first occurrence is the sign, the second is only spoken
    expect(pieces("Mer mjölk, mer mjölk!", ["mjölk"])).toEqual([
      { text: "Mer ", at: null },
      { text: "mjölk", at: 0 },
      { text: ", mer mjölk!", at: null },
    ]);
  });

  it("keeps the spoken text around them", () => {
    expect(pieces("En blå bil", ["blå"])).toEqual([
      { text: "En ", at: null },
      { text: "blå", at: 0 },
      { text: " bil", at: null },
    ]);
  });
});
