<script lang="ts">
  import { useCamera } from "$lib/camera.svelte";

  // How loud the microphone has been over the last two seconds, as a strip of bars under the
  // camera, so the learner can see they are being heard. It moves only while `Recorder` listens,
  // dim while it waits for the voice and red, like the outline, once the voice is being recorded.
  // Between listening, while an attempt is judged, it stands still on the take that was heard
  // rather than disappearing (user, 2026-09-25).
  const TICK = 50; // ms per bar, the rate `Recorder` reads the level at
  const BARS = 40; // two seconds of them

  const camera = useCamera();
  let canvas = $state<HTMLCanvasElement>();
  let heard = $state(false); // hidden until the first listening, since there is nothing to hold still
  const levels: number[] = [];

  $effect(() => {
    if (!camera.listening || !canvas) return;
    heard = true;
    const ticking = setInterval(() => {
      levels.push(camera.tracker?.level ?? 0);
      if (levels.length > BARS) levels.shift();
      draw(canvas!, levels);
    }, TICK);
    return () => clearInterval(ticking);
  });

  /** The newest bar on the right. The height is the square root of the level, so a quiet room
   * (about 0.002) still shows as a flicker and speech (above 0.05) fills half the strip or more. */
  function draw(canvas: HTMLCanvasElement, levels: number[]) {
    const ratio = devicePixelRatio;
    const [width, height] = [canvas.clientWidth * ratio, canvas.clientHeight * ratio];
    (canvas.width = width), (canvas.height = height);
    const context = canvas.getContext("2d")!;
    context.fillStyle = getComputedStyle(canvas).color;
    const step = width / BARS;
    levels.forEach((level, i) => {
      const bar = Math.max(ratio, Math.min(1, Math.sqrt(level / 0.2)) * height);
      context.fillRect(width - (levels.length - i) * step, height - bar, step * 0.6, bar);
    });
  }
</script>

<!-- the space is held before anything is heard, so the page does not move when the strip appears -->
<canvas bind:this={canvas} class:on={heard} class:recording={camera.recording}></canvas>

<style>
  canvas {
    display: block;
    width: 100%;
    height: 24px;
    margin-top: 8px;
    visibility: hidden;
    color: var(--dim);
  }

  canvas.on {
    visibility: visible;
  }

  canvas.recording {
    color: var(--bad);
  }
</style>
