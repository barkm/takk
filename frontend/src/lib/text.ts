/** Cutting a story's part into what is signed and what is only spoken.

 * The forms come from the server, which checked that each one stands in the part's text and found
 * them in the order they are spoken (`signed_words` in `story.py`). They are walked here in that
 * same order, each looked for from the end of the previous one, so a form said twice is two pieces
 * and each piece carries the index of the sign it is — which is the index of the verdict for it.
 *
 * The boundary around a form is written out rather than left to `\b`, which is ASCII-only in
 * JavaScript: `\bblå\b` matches nothing at all — "å" is not a word character, so there is no
 * boundary between it and the space after it — and a word like blå was scored by the server while
 * the page left it unmarked. */
const BOUNDARY = "[\\p{L}\\p{N}_]";

/** A piece of a part: its text, and which sign it is when it is one. `at` is null for the spoken
 * text between the signs. */
export type Piece = { text: string; at: number | null };

/** `text` cut into its pieces, one per form of `forms` and the spoken text around them. The forms
 * are matched whole and without regard to case, as the server matched them: "Mamma" opening a part
 * is the word "mamma", and "blå" is not found inside "blåbär". A form the text does not say is
 * skipped, which the server does not allow but leaves the rest of the line readable. */
export function pieces(text: string, forms: string[]): Piece[] {
  const out: Piece[] = [];
  let cursor = 0;
  forms.forEach((form, at) => {
    const escaped = form.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(`(?<!${BOUNDARY})${escaped}(?!${BOUNDARY})`, "giu");
    pattern.lastIndex = cursor;
    const found = pattern.exec(text);
    if (!found) return;
    if (found.index > cursor) out.push({ text: text.slice(cursor, found.index), at: null });
    out.push({ text: found[0], at });
    cursor = found.index + found[0].length;
  });
  if (cursor < text.length) out.push({ text: text.slice(cursor), at: null });
  return out;
}
