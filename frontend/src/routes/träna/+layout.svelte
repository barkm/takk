<script lang="ts">
  import CameraView from "$lib/Camera.svelte";
  import Level from "$lib/Level.svelte";
  import { about } from "$lib/about";
  import { useCamera } from "$lib/camera.svelte";

  // Träna is the two modes and the camera they share (user, 2026-09-23). The camera is mounted here
  // rather than in the root layout, so it runs only where something is signed, and it stays open
  // while the page moves between its start card and a mode.
  let { children } = $props();

  const camera = useCamera();
  const speaks = about()?.speaks ?? true; // the microphone's strip is only for a learner who speaks
</script>

<div class="split">
  <div class="work">{@render children()}</div>
  <aside class="camera"><CameraView {camera} />{#if speaks}<Level />{/if}</aside>
</div>

<style>
  /* What is signed on the left, what it looks like on the right, side by side on a desktop window. */
  .split {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(260px, 320px);
    gap: 28px;
    align-items: stretch; /* so a page can be as tall as the camera beside it; the camera stays at the top */
  }

  .work {
    display: flex;
    flex-direction: column;
    gap: 20px;
    min-width: 0;
  }

  .camera {
    position: sticky;
    top: 24px;
    align-self: start;
  }

  @media (max-width: 860px) {
    .split {
      grid-template-columns: 1fr;
    }

    /* the picture first, since it is what the learner watches while signing */
    .camera {
      position: static;
      order: -1;
    }
  }
</style>
