<script lang="ts">
  import { base } from "$app/paths";
import { page } from "$app/state";

  import CameraView from "$lib/Camera.svelte";
  import Level from "$lib/Level.svelte";
  import { useCamera } from "$lib/camera.svelte";
  import { hand, setHand, type Hand } from "$lib/hand";

  // Träna is the two modes and the camera they share (user, 2026-09-23). The camera is mounted here
  // rather than in the root layout, so it runs only where something is signed, and it survives the
  // switch between Ord and Meningar: a layout stays mounted while its pages come and go.
  let { children } = $props();

  const camera = useCamera();
  const path = $derived(decodeURIComponent(page.url.pathname).slice(base.length));

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
  <nav class="modes">
    <a href="{base}/träna/ord" class:on={path === "/träna/ord"}>Ord</a>
    <a href="{base}/träna/meningar" class:on={path === "/träna/meningar"}>Meningar</a>
  </nav>

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

  /* The two modes as one switch, so they read as two halves of Träna rather than two destinations. */
  .modes {
    display: inline-flex;
    gap: 2px;
    padding: 3px;
    margin-bottom: 24px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
  }

  .modes a {
    padding: 8px 22px;
    border-radius: 9px;
    color: var(--dim);
    font-weight: 500;
    text-decoration: none;
  }

  .modes a.on {
    background: var(--raised);
    color: var(--text);
  }

  /* What is signed on the left, what it looks like on the right, side by side on a desktop window. */
  .split {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(300px, 380px);
    gap: 28px;
    align-items: start;
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
