<script lang="ts">
  import { page } from "$app/state";

  import { fetchLexicon } from "$lib/api";
  import CameraView from "$lib/Camera.svelte";
  import { Camera, provideCamera } from "$lib/camera.svelte";

  import "../app.css";

  let { children } = $props();

  // The camera belongs to the app rather than to a section (user, 2026-09-22): it sits above every
  // page, the menu included, so the app looks the same throughout and the tracked skeleton is there
  // to sign at. It also opens once for the session, so no page waits for the landmarker to load.
  const camera = new Camera();
  provideCamera(camera);

  $effect(() => {
    fetchLexicon()
      .then((loaded) => (camera.lexicon = loaded))
      .catch(() => (camera.fit = "Servern svarar inte. Starta den med uv run takk."));
  });

  // One way back on every page but the menu, so nothing depends on the browser's own back button.
  // It goes to the section above rather than to wherever the learner came from: the map is a tree,
  // and a page's parent is its path without the last segment.
  const parent = $derived(page.url.pathname.replace(/\/[^/]*$/, "") || "/");
</script>

<main>
  {#if page.url.pathname !== "/"}
    <a class="back" href={parent}>Tillbaka</a>
  {/if}
  <CameraView {camera} />
  {@render children()}
</main>

<style>
  .back {
    align-self: flex-start;
    color: var(--dim);
    text-decoration: none;
  }

  .back::before {
    content: "← ";
  }
</style>
