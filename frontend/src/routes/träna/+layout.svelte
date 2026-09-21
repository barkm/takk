<script lang="ts">
  import { fetchLexicon } from "$lib/api";
  import CameraView from "$lib/Camera.svelte";
  import { Camera, provideCamera } from "$lib/camera.svelte";

  // Träna (step 12 of ROADMAP-takk.md): the camera belongs to the section, not to either mode. This
  // layout stays mounted while the hub, Nya ord and Repetera come and go under it, so the camera
  // opens once, keeps running between the modes, and the element the tracker draws on stays put. The
  // picture itself is the shared component, which Sök mounts the same way.
  let { children } = $props();

  const camera = new Camera();
  provideCamera(camera);

  $effect(() => {
    fetchLexicon()
      .then((loaded) => (camera.lexicon = loaded))
      .catch(() => (camera.fit = "Servern svarar inte. Starta den med uv run takk."));
  });
</script>

{#if camera.lexicon}
  <CameraView {camera} edges={camera.lexicon.edges} />
{:else}
  <p class="dim">{camera.fit || "Laddar lexikonet ..."}</p>
{/if}

{@render children()}
