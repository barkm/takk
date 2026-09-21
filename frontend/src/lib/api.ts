// The API is a separate deployment (see ROADMAP-takk.md), so its origin is configurable. Empty in
// development and in `npm run preview`, where Vite proxies /api to the local server.
import { resample, type Edges, type Frame } from "$lib/landmarks";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export const api = (path: string) => `${BASE}${path}`;

/** A sign of an attempt: the sign that scores it, its lexicon clips, and the word the signer says
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

/** A word to practise: the sign that scores it, and the word to show when the sign is not named for
 * it (entries sharing a sign form are one class labelled by its lowest entry, so "äta" is scored as
 * `sts:livsmedel-01265`). */
export type PackWord = { sign: string; id: string; word?: string };

/** A named list of words the learner can turn on: a starter pack or one of the lexicon's categories. */
export type Pack = { name: string; kind: "pack" | "category"; words: PackWord[] };

export async function fetchPacks(): Promise<Pack[]> {
  const response = await fetch(api("/api/packs"));
  return (await response.json()).packs;
}

/** A Swedish sentence whose key words are `words`, to be spoken while they are signed, with the words
 * in the order they occur in it. The sentence is empty when the server could not write one, and the
 * caller then practises the words one at a time. The words are the ones the learner is practising,
 * not the names of the signs that score them (see `Sign.spoken`). */
export async function fetchSentence(words: string[]): Promise<{ sentence: string; words: string[] }> {
  const response = await fetch(api("/api/sentence"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ words }),
  });
  if (!response.ok) return { sentence: "", words: [] };
  return await response.json();
}

/** A part of a story: the Swedish text to say aloud, and the words of it that are signed, in the
 * order they are spoken. */
export type StoryPart = { text: string; words: string[] };

/** A Swedish story over `words` in `parts` parts, one recording each (see `story.py`). Empty when the
 * server could not write one. The words are the learner's own, as in `fetchSentence`. */
export async function fetchStory(words: string[], parts: number): Promise<StoryPart[]> {
  const response = await fetch(api("/api/story"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ words, parts }),
  });
  if (!response.ok) return [];
  return (await response.json()).parts;
}

/** How the lexicon describes the form of an entry's sign, in Swedish. Empty when it describes none. */
export async function fetchForm(entryId: string): Promise<string> {
  const response = await fetch(api(`/api/form/${encodeURIComponent(entryId)}`));
  return (await response.json()).form;
}

export const referenceUrl = (clip: string) => api(`/api/reference/${encodeURIComponent(clip)}`);

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

/** The Swedish word a lexicon sign is signed for, and its entry number ("sts:platta slag-25563"). */
export const word = (sign: string) => sign.replace(/^sts:/, "").replace(/-\d+$/, "");
export const entry = (sign: string) => sign.match(/\d+$/)?.[0] ?? "";
