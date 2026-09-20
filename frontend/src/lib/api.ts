// The API is a separate deployment (see ROADMAP-takk.md), so its origin is configurable. Empty in
// development and in `npm run preview`, where Vite proxies /api to the local server.
import { resample, type Edges, type Frame } from "$lib/landmarks";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export const api = (path: string) => `${BASE}${path}`;

export type Sign = { sign: string; references: string[] };

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
  /** How the sentence was split: by the spoken words, or not at all when it is a single sign. */
  split: "speech" | "whole";
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

/** A Swedish sentence whose key words are `signs`, to be spoken while they are signed, with the signs
 * in the order they occur in it. The sentence is empty when the server could not write one, and the
 * caller then practises the signs one at a time. */
export async function fetchSentence(signs: string[]): Promise<{ sentence: string; signs: string[] }> {
  const response = await fetch(api("/api/sentence"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ signs }),
  });
  if (!response.ok) return { sentence: "", signs: [] };
  return await response.json();
}

/** How the lexicon describes the form of an entry's sign, in Swedish. Empty when it describes none. */
export async function fetchForm(entryId: string): Promise<string> {
  const response = await fetch(api(`/api/form/${encodeURIComponent(entryId)}`));
  return (await response.json()).form;
}

export const referenceUrl = (clip: string) => api(`/api/reference/${encodeURIComponent(clip)}`);

/** Score a recording as an attempt of `signs`, in order. Throws when the server refuses it. */
export async function scoreAttempt(
  frames: Frame[],
  audio: Blob | null,
  audioStart: number,
  signs: string[],
  handedness: "left" | "right",
  video: HTMLVideoElement,
  fps: number,
): Promise<Attempt> {
  const body = new FormData();
  const landmarks = resample(frames, fps);
  body.append("landmarks", new Blob([landmarks.buffer as ArrayBuffer]), "landmarks.f32");
  if (audio) {
    body.append("audio", audio, "audio");
    body.append("audio_offset", String(frames[0].time - audioStart));
  }
  for (const sign of signs) body.append("sign", sign);
  body.append("handedness", handedness);
  body.append("width", String(video.videoWidth));
  body.append("height", String(video.videoHeight));
  const response = await fetch(api("/api/attempt"), { method: "POST", body });
  if (!response.ok) throw new Error(`the server answered ${response.status}`);
  return response.json();
}

/** The Swedish word a lexicon sign is signed for, and its entry number ("sts:platta slag-25563"). */
export const word = (sign: string) => sign.replace(/^sts:/, "").replace(/-\d+$/, "");
export const entry = (sign: string) => sign.match(/\d+$/)?.[0] ?? "";
