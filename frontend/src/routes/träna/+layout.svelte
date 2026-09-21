<script lang="ts">
  import { fetchLexicon } from "$lib/api";
  import { Camera, provideCamera } from "$lib/camera.svelte";
  import { framing } from "$lib/framing";
  import { hand } from "$lib/hand";
  import { Tracker } from "$lib/tracking";

  // Träna (step 12 of ROADMAP-takk.md): the camera belongs to the section, not to either mode. This
  // layout stays mounted while the hub, Nya ord and Repetera come and go under it, so the camera
  // opens once, keeps running between the modes, and the element the tracker draws on stays put.
  const LOOK = 200; // ms between readings of how the signer sits; faster than that only flickers

  let { children } = $props();

  const camera = new Camera();
  provideCamera(camera);

  let video = $state<HTMLVideoElement>();
  let canvas = $state<HTMLCanvasElement>();
  let starting: Promise<unknown> | null = null;

  $effect(() => {
    camera.handedness = hand() ?? "right"; // asked once on the menu, never on a camera screen
    fetchLexicon()
      .then((loaded) => (camera.lexicon = loaded))
      .catch(() => (camera.fit = "Servern svarar inte. Starta den med uv run takk."));
  });

  $effect(() => {
    if (!camera.lexicon || !video || !canvas) return;
    camera.video = video;
    starting ??= Tracker.start(video, canvas, camera.lexicon.edges)
      .then((started) => (camera.tracker = started))
      .catch((error) => (camera.fit = `Ingen åtkomst till kameran: ${error.message}`));
  });

  // How the signer sits, read off the frame being tracked right now, so the framing is fixed before a
  // recording is spent on it. While recording it also asks for the hands.
  $effect(() => {
    if (!camera.tracker) return;
    const looking = setInterval(() => {
      const latest = camera.tracker?.latest;
      camera.fit = latest ? framing(latest, camera.recording) : "";
    }, LOOK);
    return () => clearInterval(looking);
  });
</script>

<div class="view" class:recording={camera.recording}>
  <!-- svelte-ignore a11y_media_has_caption -->
  <video bind:this={video} autoplay muted playsinline></video>
  <canvas bind:this={canvas}></canvas>
  {#if camera.fit}<p class="fit">{camera.fit}</p>{/if}
</div>

{@render children()}

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

  .view.recording video {
    outline: 3px solid var(--bad);
  }

  canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  /* over the picture it is about, and unmirrored: the view itself is flipped like a mirror */
  .fit {
    position: absolute;
    inset: auto 0 0;
    margin: 0;
    padding: 8px;
    transform: scaleX(-1);
    text-align: center;
    background: #0009;
    border-radius: 0 0 8px 8px;
  }
</style>
