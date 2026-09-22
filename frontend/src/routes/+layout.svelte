<script lang="ts">
  import { page } from "$app/state";

  import "../app.css";

  let { children } = $props();

  // One way back on every page but the menu, so nothing depends on the browser's own back button.
  // It goes to the section above rather than to wherever the learner came from: the map is a tree,
  // and a page's parent is its path without the last segment.
  const parent = $derived(page.url.pathname.replace(/\/[^/]*$/, "") || "/");
</script>

<main>
  {#if page.url.pathname !== "/"}
    <a class="back" href={parent}>Tillbaka</a>
  {/if}
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
