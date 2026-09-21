/** Cutting a story's part into what is signed and what is only spoken.

 * The key words come from the server, which found them with Python's `re` and its unicode word
 * boundaries (`key_words` in `story.py`). JavaScript's `\b` is ASCII-only, so `\bblå\b` matches
 * nothing at all — "å" is not a word character, so there is no boundary between it and the space
 * after it — and a word like blå was scored by the server while the page left it unmarked. The
 * boundary is therefore written out as letters, digits and underscore in any script. */
const BOUNDARY = "[\\p{L}\\p{N}_]";

/** `text` cut into its pieces, each said whether it is one of `words` (a `key` piece) or the spoken
 * text between them. The words are matched whole and without regard to case, as the server matched
 * them: "Mamma" opening a part is the word "mamma", and "blå" is not found inside "blåbär". */
export function pieces(
  text: string,
  words: string[],
): { text: string; key: boolean }[] {
  if (!words.length) return [{ text, key: false }];
  const pattern = words
    .map((each) => each.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  const split = text.split(
    new RegExp(`(?<!${BOUNDARY})(${pattern})(?!${BOUNDARY})`, "iu"),
  );
  return split.map((each, index) => ({ text: each, key: index % 2 === 1 }));
}
