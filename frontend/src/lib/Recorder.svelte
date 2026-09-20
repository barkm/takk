<script lang="ts">
  import { scoreAttempt, type Attempt, type Lexicon, type Sign } from "$lib/api";
  import { Tracker } from "$lib/tracking";

  let {
    sentence,
    lexicon,
    onattempt,
  }: { sentence: Sign[]; lexicon: Lexicon; onattempt: (attempt: Attempt | null, note: string) => void } = $props();

  const SLOW_FPS = 20; // below this the extractor skips so many camera frames that results suffer
  const NO_MICROPHONE = "Mikrofonen behövs: orden du säger högt är det som visar var tecknen är i inspelningen.";

  let video: HTMLVideoElement;
  let canvas: HTMLCanvasElement;
  let tracker = $state<Tracker | null>(null); // the generic, as in the pages: an annotation narrows it to null
  let starting: Promise<Tracker> | null = null;
  let status = $state("Laddar teckenmodellen...");
  let handedness: "left" | "right" = $state("right");
  let recording = $state(false);
  let scoring = $state(false);
  let seconds = $state(0);
  let ticker: ReturnType<typeof setInterval> | null = null;

  // The camera starts once, however fast signs are picked, and keeps running between attempts.
  $effect(() => {
    starting ??= Tracker.start(video, canvas, lexicon.edges, (fps) => {
      const slow = fps < SLOW_FPS ? " — långsammare än kameran, så resultatet blir mindre tillförlitligt" : "";
      status = `Följer tecknen live på ${tracker?.delegate} i ${fps.toFixed(0)} fps${slow}`;
    })
      .then((started) => (tracker = started))
      .catch((error) => {
        status = `Ingen åtkomst till kameran: ${error.message}`;
        throw error;
      });
  });

  // Every attempt is located by the words the signer says, so without the microphone there is
  // nothing to locate it with. Refusing here says so before a recording is made and thrown away.
  const silent = $derived(tracker?.hasAudio === false);

  function start() {
    if (!tracker || silent) return;
    onattempt(null, "");
    tracker.startRecording();
    recording = true;
    const started = Date.now();
    ticker = setInterval(() => {
      seconds = (Date.now() - started) / 1000;
      if (seconds > lexicon.maxSeconds * sentence.length) stop();
    }, 100);
  }

  async function stop() {
    if (!tracker || !recording) return; // the time limit and the button can both stop a recording
    recording = false;
    if (ticker) clearInterval(ticker);
    const taken = await tracker.stopRecording();
    if (!taken) return;
    if (taken.frames.length < 2) return onattempt(null, "Inspelningen är tom.");
    scoring = true;
    try {
      const attempt = await scoreAttempt(
        taken.frames,
        taken.audio,
        taken.audioStart,
        sentence,
        handedness,
        video,
        lexicon.fps,
      );
      onattempt(attempt, "");
    } catch {
      onattempt(null, "Bedömningen misslyckades. Spela in igen.");
    } finally {
      scoring = false;
    }
  }

  // Space toggles recording; preventDefault also stops it clicking whichever button has focus.
  function onkeydown(event: KeyboardEvent) {
    if (event.code !== "Space" || (event.target as HTMLInputElement).type === "search") return;
    event.preventDefault();
    if (!event.repeat && tracker && !scoring) (recording ? stop() : start());
  }
</script>

<svelte:window {onkeydown} />

<section class="card">
  <div class="view">
    <!-- svelte-ignore a11y_media_has_caption -->
    <video bind:this={video} class:recording autoplay muted playsinline></video>
    <canvas bind:this={canvas}></canvas>
  </div>
  <p class="dim">{status}</p>
  {#if silent}<p class="dim">{NO_MICROPHONE}</p>{/if}
  <p class="row">
    <button disabled={!tracker || scoring || silent} onclick={() => (recording ? stop() : start())}>
      {recording ? "Stoppa" : "Spela in"}
    </button>
    <span class="dim">{recording ? `${seconds.toFixed(1)} s` : ""}</span>
    <label class="row dim">
      Jag tecknar med
      <input type="radio" name="handedness" value="right" bind:group={handedness} /> höger
      <input type="radio" name="handedness" value="left" bind:group={handedness} /> vänster hand
    </label>
  </p>
  {#if scoring}
    <p class="dim">Bedömer...</p>
  {/if}
</section>

<style>
  .view {
    position: relative;
    width: 100%;
    max-width: 560px;
    transform: scaleX(-1);
  }

  video {
    display: block;
    width: 100%;
    border-radius: 8px;
    background: #000;
  }

  video.recording {
    outline: 3px solid var(--bad);
  }

  canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
</style>
