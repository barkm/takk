// The landmarker runs in this worker rather than on the page: setting it up compiles its model for
// about as long as the camera takes to start, and on the page's own thread that froze the whole app
// for those seconds, so not even the sections could be switched (user, 2026-09-26). The page sends
// camera frames as they come and gets each frame's landmarks back in the common layout.
import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision";

import { layout } from "$lib/landmarks";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
// The landmarker is loaded from MediaPipe's own address, as the wasm beside it is, so neither the
// API nor this site carries the 13.7 MB (see ROADMAP-takk.md). It is the model extraction.py
// downloads, by the same URL.
const MODEL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";

// MediaPipe's wasm loader is a classic script that sets a global, which a module worker can neither
// `importScripts` nor `import`. MediaPipe loads it through `self.import` when there is one, so this
// runs it in the worker's global scope instead.
Object.assign(self, { import: async (url: string) => (0, eval)(await (await fetch(url)).text()) });

async function load(): Promise<HolisticLandmarker> {
  const files = await FilesetResolver.forVisionTasks(WASM);
  const create = (delegate: "GPU" | "CPU") =>
    HolisticLandmarker.createFromOptions(files, { baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: "VIDEO" });
  return create("GPU").catch(() => create("CPU")); // the CPU if the GPU is unavailable
}

const loading = load();
loading.then(
  () => postMessage({ ready: true }),
  (error) => postMessage({ error: String(error?.message ?? error) }),
);

onmessage = async ({ data: { frame, time } }: MessageEvent<{ frame: ImageBitmap; time: number }>) => {
  const landmarks = layout((await loading).detectForVideo(frame, time));
  frame.close();
  postMessage({ landmarks }, { transfer: [landmarks.buffer] });
};
