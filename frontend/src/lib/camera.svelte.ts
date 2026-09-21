// The camera of the practice modes, which belongs to the Träna section rather than to either mode
// (step 12 of ROADMAP-takk.md). The layout owns the video element, the landmarker and the framing
// feedback, and the pages under it reach them through this context: a SvelteKit layout stays mounted
// while its pages come and go, so the camera opens once and the element the tracker draws on is never
// taken out from under it.
import { getContext, setContext } from "svelte";

import type { Lexicon } from "$lib/api";
import type { Tracker } from "$lib/tracking";

export class Camera {
  tracker = $state<Tracker | null>(null);
  video = $state<HTMLVideoElement>();
  lexicon = $state<Lexicon | null>(null);
  /** What to fix about the framing, or FRAMING.ok; a camera that failed to open says so here. */
  fit = $state("");
  /** Which hand the signer signs with, which is what a recording is mirrored by. */
  handedness = $state<"left" | "right">("right");
  /** Whether a recording is running, which the framing feedback also asks for the hands during. */
  recording = $state(false);
}

const KEY = Symbol("camera");

export const provideCamera = (camera: Camera) => setContext(KEY, camera);

export const useCamera = () => getContext<Camera>(KEY);
