<script lang="ts">
  import { framing } from "$lib/framing";
  import { hand } from "$lib/hand";
  import { Tracker } from "$lib/tracking";
  import type { Camera } from "$lib/camera.svelte";
  import type { Edges } from "$lib/landmarks";

  // The camera picture with the tracked landmarks drawn over it, and what is wrong with the framing
  // written over the bottom of it. Both places that show a camera use this (steps 10 to 12 of
  // ROADMAP-takk.md): Sök, to look a sign up by signing it, and the Träna layout, which keeps it
  // mounted while the practice modes come and go under it.
  //
  // The camera closes when this component goes away. Nothing else stops the stream, so leaving the
  // page would otherwise leave the camera light on and the next page would open a second stream.
  const LOOK = 200; // ms between readings of how the signer sits; faster than that only flickers

  let { camera, edges, children }: { camera: Camera; edges: Edges; children?: import("svelte").Snippet } = $props();

  let video = $state<HTMLVideoElement>();
  let canvas = $state<HTMLCanvasElement>();
  let starting: Promise<unknown> | null = null;

  $effect(() => {
    if (!video || !canvas) return;
    camera.video = video;
    camera.handedness = hand() ?? "right"; // asked once on the menu, never on a camera screen
    starting ??= Tracker.start(video, canvas, edges)
      .then((started) => (camera.tracker = started))
      .catch((error) => (camera.fit = `Ingen åtkomst till kameran: ${error.message}`));
    return () => {
      camera.tracker?.stop();
      (camera.tracker = null), (starting = null);
    };
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

<div class="view" class:recording={camera.recording}>
  <!-- svelte-ignore a11y_media_has_caption -->
  <video bind:this={video} autoplay muted playsinline></video>
  <canvas bind:this={canvas}></canvas>
  {@render children?.()}
  {#if camera.fit}<p class="fit">{camera.fit}</p>{/if}
</div>

<style>
  .view {
    position: relative;
    width: 100%;
    max-width: 560px;
    transform: scaleX(-1); /* a mirror, which is how a signer expects to see themselves */
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
