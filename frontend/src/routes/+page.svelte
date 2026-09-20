<script lang="ts">
  import { api } from "$lib/api";

  // A placeholder page: it only shows that the build works and that the API answers. The practice UI
  // is ported from web/practice.html next.
  let signs = $state<number | null>(null);
  let failed = $state(false);

  $effect(() => {
    fetch(api("/api/signs"))
      .then((response) => response.json())
      .then((data) => (signs = data.signs.length))
      .catch(() => (failed = true));
  });
</script>

<h1>Practice a sign or a sentence</h1>
{#if signs !== null}
  <p>{signs} signs in the lexicon.</p>
{:else if failed}
  <p>The server does not answer. Start it with <code>uv run takk</code>.</p>
{:else}
  <p>Loading the lexicon …</p>
{/if}
