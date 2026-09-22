// The camera of the whole app, which the root layout owns and every page reaches through this
// context (step 12 of ROADMAP-takk.md). A SvelteKit layout stays mounted while the pages under it
// come and go, so the camera opens once for the session, the element the tracker draws on is never
// taken out from under it, and no page pays for starting it.
import { getContext, setContext, type Snippet } from "svelte";

import type { Lexicon } from "$lib/api";
import type { Tracker } from "$lib/tracking";

export class Camera {
  tracker = $state<Tracker | null>(null);
  video = $state<HTMLVideoElement>();
  lexicon = $state<Lexicon | null>(null);
  /** What to fix about the framing, or FRAMING.ok; a camera that failed to open says so here. */
  fit = $state("");
  /** What a page puts in the picture over the live one, such as Sök's replay of the sign it
   * searched with. The picture belongs to the app rather than to a page, so a page hands in what it
   * wants shown in it instead of wrapping it. */
  overlay = $state<Snippet | null>(null);
  /** Whether a recording is running, which the framing feedback also asks for the hands during. */
  recording = $state(false);
}

const KEY = Symbol("camera");

export const provideCamera = (camera: Camera) => setContext(KEY, camera);

export const useCamera = () => getContext<Camera>(KEY);
