// Who the learner is, asked on Om dig the first time the app is opened and kept in the browser
// (step 23 of ROADMAP-takk.md): whether they have signed before, and in their own words who they
// sign with and where, and whether they speak while signing. The model reads it for the chips on Sök and the text under Meningar; the
// hand is kept apart in `hand.ts`, since a recording needs it and nothing else does.
export type Level = "ny" | "lite" | "van";
/** `speaks` is false for a learner who signs in silence: their signs are then told apart by the
 * hands being lowered between them rather than by the words said over them (`Recorder.svelte`). */
export type About = { level: Level; context: string; speaks: boolean };

const KEY = "takk.about";

/** What the learner said about themselves, or null until they have been asked. */
export function about(): About | null {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "null");
  } catch {
    return null; // an unreadable store asks again, as the hand does
  }
}

export function setAbout(said: About) {
  localStorage.setItem(KEY, JSON.stringify(said));
}
