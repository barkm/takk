<script lang="ts">
  import { framing } from "$lib/framing";
  import { Tracker } from "$lib/tracking";
  import type { Camera } from "$lib/camera.svelte";

  // The camera picture with the tracked landmarks drawn over it, and what is wrong with the framing
  // written over the bottom of it. The root layout mounts it above every page (step 12 of
  // ROADMAP-takk.md), so the app looks the same throughout and the moving skeleton invites signing
  // even on the menu, where nothing reads it.
  //
  // The camera closes when this component goes away, which is when the app is left. Nothing else
  // stops the stream, so it would otherwise leave the camera light on.
  const LOOK = 200; // ms between readings of how the signer sits; faster than that only flickers

  let { camera }: { camera: Camera } = $props();

  let video = $state<HTMLVideoElement>();
  let canvas = $state<HTMLCanvasElement>();
  let starting: Promise<unknown> | null = null;

  $effect(() => {
    if (!video || !canvas) return;
    camera.video = video;
    starting ??= Tracker.start(video, canvas)
      .then((started) => (camera.tracker = started))
      .catch((error) => (camera.fit = `Ingen åtkomst till kameran: ${error.message}`));
    return () => {
      camera.tracker?.stop();
      (camera.tracker = null), (starting = null);
    };
  });

  // The lines between the landmarks come from the lexicon, which arrives after the camera has
  // started, so the tracker is given them when they are there.
  $effect(() => {
    if (camera.tracker) camera.tracker.edges = camera.lexicon?.edges ?? {};
  });

  // How the signer sits, read off the frame being tracked right now, so the framing is fixed before a
  // recording is spent on it. It says nothing about the hands, which come and go with every sign.
  $effect(() => {
    if (!camera.tracker) return;
    const looking = setInterval(() => {
      const latest = camera.tracker?.latest;
      camera.fit = latest ? framing(latest) : "";
    }, LOOK);
    return () => clearInterval(looking);
  });
</script>

<div class="view" class:recording={camera.recording} class:ready={camera.tracker}>
  <!-- svelte-ignore a11y_media_has_caption -->
  <video bind:this={video} autoplay muted playsinline></video>
  <canvas bind:this={canvas}></canvas>
  {#if camera.tracker}
    {@render camera.overlay?.()}
    {#if camera.fit}<p class="fit">{camera.fit}</p>{/if}
  {:else}
    <!-- the space is reserved above, so only what fills it changes when the camera is ready -->
    <p class="starting">{camera.fit || "Startar kameran …"}</p>
  {/if}
</div>

<style>
  /* the box the picture will fill, held open at the shape of the camera from the first frame of the
     page, so nothing jumps when the camera and the landmark model are finally ready */
  .view {
    position: relative;
    width: 100%;
    aspect-ratio: 4 / 3;
    background: #000;
    border: 1px solid var(--line);
    border-radius: var(--radius);
    overflow: hidden;
    transform: scaleX(-1); /* a mirror, which is how a signer expects to see themselves */
  }

  /* the picture stays out of sight until the landmarks can be drawn over it, so the feed and the
     tracking appear together rather than the raw camera first */
  .view:not(.ready) video,
  .view:not(.ready) canvas {
    visibility: hidden;
  }

  video {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  /* a recording is marked on the picture itself, since that is where the learner is looking */
  .view.recording {
    border-color: var(--bad);
    box-shadow: 0 0 0 1px var(--bad);
  }

  canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  /* what stands in for the picture until it is there, unmirrored like the framing text below */
  .starting {
    position: absolute;
    inset: 0;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    transform: scaleX(-1);
    text-align: center;
    color: var(--dim);
    font-size: 14px;
  }

  /* over the picture it is about, and unmirrored: the view itself is flipped like a mirror */
  .fit {
    position: absolute;
    inset: auto 0 0;
    margin: 0;
    padding: 10px;
    transform: scaleX(-1);
    text-align: center;
    font-size: 14px;
    background: #000000b3;
  }
</style>
