// The API is a separate deployment (see ROADMAP-takk.md), so its origin is configurable. Empty in
// development and in `npm run preview`, where Vite proxies /api to the local server.
import { resample, type Edges, type Frame } from "$lib/landmarks";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export const api = (path: string) => `${BASE}${path}`;

/** A sign of an attempt: the sign that scores it, the addresses of its lexicon clips (the lexicon's
 * own, so they play whether or not the API is up), and the word the signer says
 * for it when that is not the sign's own name (a learner practising "blå" signs `sts:öga-02636`,
 * because blå and öga are one sign form and the lower entry names the class). */
export type Sign = { sign: string; references: string[]; spoken?: string };

export type Lexicon = {
  signs: Sign[];
  /** The frame rate a recording is resampled to before it is sent. */
  fps: number;
  /** The longest recording accepted per sign of the sentence. */
  maxSeconds: number;
  edges: Edges;
};

/** One sign of an attempt, as the server judged it. */
export type Judgement = {
  sign: string;
  usable: boolean;
  note: string;
  score?: number;
  correct?: boolean;
  closest?: { sign: string; score: number };
};

export type Attempt = {
  threshold: number;
  /** Empty when the recording could not be split into the sentence's signs; `note` says why. */
  signs: Judgement[];
  note: string;
};

export async function fetchLexicon(): Promise<Lexicon> {
  const response = await fetch(api("/api/signs"));
  const data = await response.json();
  return { signs: data.signs, fps: data.fps, maxSeconds: data.max_seconds, edges: data.edges };
}

/** A word to learn: the sign that scores it, and the word to show when the sign is not named for
 * it (entries sharing a sign form are one class labelled by its lowest entry, so "äta" is scored as
 * `sts:livsmedel-01265`). */
export type SignWord = { sign: string; id: string; word?: string };

/** The lexicon's own page for an entry, linked from Sök and Tecken but not from the practice pages,
 * where it would show the answer. */
export const lexiconUrl = (id: string) => `https://teckensprakslexikon.su.se/ord/${id}`;

/** The signs a learner searching for `query` is offered, a word or a theme alike. This is the one way
 * vocabulary grows (step 9 of ROADMAP-takk.md): what is ticked here is what Ord teaches. */
export async function fetchSearch(query: string): Promise<SignWord[]> {
  const response = await fetch(api(`/api/search?q=${encodeURIComponent(query)}`));
  if (!response.ok) return [];
  return (await response.json()).words;
}

/** A part of a story: the Swedish text to say aloud, and its signs in the order they are spoken.
 * `said` is the form as the text says it, which is what is spoken and marked; `word` is the word of
 * the learner's own list that the form signs, which is what names the sign and moves its box. */
export type StoryPart = { text: string; signs: { said: string; word: string }[] };

/** A Swedish story over `words` in `parts` parts, one recording each (see `story.py`), set in the
 * everyday life the learner described on Om dig (`about`). Empty when the server could not write
 * one. The words are the learner's own, as in `fetchSentence`. */
export async function fetchStory(words: string[], parts: number, about = ""): Promise<StoryPart[]> {
  const response = await fetch(api("/api/story"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ words, parts, about }),
  });
  if (!response.ok) return [];
  return (await response.json()).parts;
}

/** A chip on Sök: a label and the words it fills the grid with (see `suggest.py`). */
export type Chip = { label: string; words: SignWord[] };

/** The chips fitted to what the learner said on Om dig, leaving out the words in `known`. Empty when
 * the server could not suggest any. */
export async function fetchChips(level: string, about: string, known: string[]): Promise<Chip[]> {
  const response = await fetch(api("/api/suggest"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ level, about, known }),
  });
  if (!response.ok) return [];
  return (await response.json()).chips;
}

/** How the lexicon describes the form of an entry's sign, in Swedish. Empty when it describes none. */
export async function fetchForm(entryId: string): Promise<string> {
  const response = await fetch(api(`/api/form/${encodeURIComponent(entryId)}`));
  return (await response.json()).form;
}

/** Score a recording as an attempt of `signs`, in order. Every attempt is spoken and its signs are
 * located by the words said over it, so it needs the microphone. Throws when the server refuses it.
 *
 * `unheardIsMiss` scores a word that was not said as a miss of its sign instead of refusing the whole
 * recording, which is what a story does: it never stops, and a word left unsaid is a word unsigned. */
export async function scoreAttempt(
  frames: Frame[],
  audio: Blob | null,
  audioStart: number,
  signs: Sign[],
  handedness: "left" | "right",
  video: HTMLVideoElement,
  fps: number,
  unheardIsMiss = false,
): Promise<Attempt> {
  const body = new FormData();
  const landmarks = resample(frames, fps);
  body.append("landmarks", new Blob([landmarks.buffer as ArrayBuffer]), "landmarks.f32");
  if (audio) {
    body.append("audio", audio, "audio");
    body.append("audio_offset", String(frames[0].time - audioStart));
  }
  for (const each of signs) body.append("sign", each.sign);
  // a word per sign, or none at all: the server then listens for each sign under its own name
  if (signs.some((each) => each.spoken)) for (const each of signs) body.append("spoken", each.spoken ?? word(each.sign));
  body.append("handedness", handedness);
  body.append("width", String(video.videoWidth));
  body.append("height", String(video.videoHeight));
  if (unheardIsMiss) body.append("unheard_is_miss", "true");
  const response = await fetch(api("/api/attempt"), { method: "POST", body });
  if (!response.ok) throw new Error(`the server answered ${response.status}`);
  return response.json();
}

/** The Swedish word a lexicon sign is signed for ("sts:platta slag-25563" -> "platta slag"). */
export const word = (sign: string) => sign.replace(/^sts:/, "").replace(/-\d+$/, "");

/** The lexicon signs closest to a recording of one sign, the nearest first (step 10 of
 * ROADMAP-takk.md). A lookup and not an attempt: nothing is spoken and nothing is scored, so the
 * answer is the same list of words a text search gives, with a note when the recording was unusable. */
export async function searchBySign(
  frames: Frame[],
  handedness: "left" | "right",
  video: HTMLVideoElement,
  fps: number,
): Promise<{ words: SignWord[]; note: string }> {
  const body = new FormData();
  const landmarks = resample(frames, fps);
  body.append("landmarks", new Blob([landmarks.buffer as ArrayBuffer]), "landmarks.f32");
  body.append("handedness", handedness);
  body.append("width", String(video.videoWidth));
  body.append("height", String(video.videoHeight));
  const response = await fetch(api("/api/search"), { method: "POST", body });
  if (!response.ok) throw new Error(`the server answered ${response.status}`);
  return response.json();
}
