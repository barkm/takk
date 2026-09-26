// How the signer sits in front of the camera, judged on the frame being tracked right now (step 11
// of ROADMAP-takk.md). The server runs `checks.py` on a finished recording and says why it could not
// be used; this says the same kind of thing before a recording is spent, so the framing is fixed
// first. It is deliberately the cheap half: one frame, a handful of distances, no history.
//
// The numbers are all knobs and none is measured. They are read off the lexicon's own framing, where
// the signer sits with head and upper body in view and room above the head for the signs made there.
import { N_LANDMARKS } from "$lib/landmarks";

const POSE = 489; // LANDMARK_SLICES in landmarks.py
const NOSE = POSE;
const SHOULDERS = [POSE + 11, POSE + 12];

/** How wide the shoulders may be as a share of the frame width, which is how far away the signer is. */
export const NARROW = 0.16;
export const WIDE = 0.55;
/** How much of the frame height has to be left above the head, for the signs made up there. */
export const HEADROOM = 0.08;
/** How far down the frame the shoulders may sit, so the upper body below them is in view. */
export const LOW = 0.5;
/** How far the shoulders' middle may sit from the middle of the frame. */
export const OFF_CENTRE = 0.18;

export const FRAMING = {
  none: "Ingen syns i bild.",
  shoulders: "Sätt dig så att båda axlarna syns.",
  near: "Sätt dig närmare kameran.",
  far: "Sätt dig längre från kameran.",
  headroom: "Rikta kameran lägre — det behövs plats ovanför huvudet.",
  low: "Rikta kameran högre — mer av överkroppen behöver synas.",
  centre: "Flytta dig mot mitten av bilden.",
};

/** What to fix about the framing, in Swedish, one thing at a time and the most basic first, and the
 * empty string when there is nothing to fix: a framing that is already right is not worth a line of
 * its own (user, 2026-09-21), and what is wrong is shown over the picture it is about.
 *
 * Only the framing, never the signing: the hands are out of view between signs and a camera that
 * asked for them was complaining about the pause (user, 2026-09-22). The server still refuses a
 * recording with no hands in it, which is where that belongs. */
export function framing(landmarks: Float32Array): string {
  if (landmarks.length !== N_LANDMARKS * 3) return FRAMING.none;
  const x = (point: number) => landmarks[3 * point];
  const y = (point: number) => landmarks[3 * point + 1];
  const [left, right] = SHOULDERS;
  if (Number.isNaN(x(left)) || Number.isNaN(x(right))) return FRAMING.shoulders;
  const width = Math.abs(x(left) - x(right));
  if (width < NARROW) return FRAMING.near;
  if (width > WIDE) return FRAMING.far;
  if (Math.abs((x(left) + x(right)) / 2 - 0.5) > OFF_CENTRE) return FRAMING.centre;
  // the nose stands in for the head, since the face is tracked as a whole or not at all
  if (Number.isNaN(y(NOSE)) || y(NOSE) < HEADROOM) return FRAMING.headroom;
  if ((y(left) + y(right)) / 2 > LOW) return FRAMING.low;
  return "";
}

