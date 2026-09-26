<script lang="ts">
  import CameraView from "$lib/Camera.svelte";
  import Level from "$lib/Level.svelte";
  import { useCamera } from "$lib/camera.svelte";
  import { hand, setHand, type Hand } from "$lib/hand";

  // Träna is the two modes and the camera they share (user, 2026-09-23). The camera is mounted here
  // rather than in the root layout, so it runs only where something is signed, and it stays open
  // while the page moves between its start card and a mode.
  let { children } = $props();

  const camera = useCamera();

  // The hand is asked for once, before the first recording, since every recording is mirrored by it.
  // It is asked here because Träna is where recordings are made; Tecken carries the switch after.
  let signs = $state<Hand | null>(null);
  let asked = $state(false);

  $effect(() => {
    (signs = hand()), (asked = true);
  });

  function choose(which: Hand) {
    setHand(which);
    signs = which;
  }
</script>

{#if !asked}
  <!-- the store is read in an effect, so nothing is shown until it is known which screen this is -->
{:else if !signs}
  <section class="card gate">
    <h1>Vilken hand tecknar du med?</h1>
    <p class="dim">Inspelningen speglas efter den. Du kan byta under Tecken.</p>
    <div class="row">
      <button onclick={() => choose("right")}>Höger</button>
      <button class="secondary" onclick={() => choose("left")}>Vänster</button>
    </div>
  </section>
{:else}
  <div class="split">
    <div class="work">{@render children()}</div>
    <aside class="camera"><CameraView {camera} /><Level /></aside>
  </div>
{/if}

<style>
  .gate {
    max-width: 480px;
  }

  .gate h1 {
    font-size: 24px;
  }

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
