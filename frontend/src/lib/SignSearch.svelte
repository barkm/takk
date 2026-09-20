<script lang="ts">
  import { word, entry, type Sign } from "$lib/api";

  let { signs, onpick }: { signs: Sign[]; onpick: (sign: Sign) => void } = $props();

  let query = $state("");

  // Signs whose word starts with the query first, then the ones that only contain it.
  const found = $derived.by(() => {
    const text = query.trim().toLowerCase();
    if (!text) return [];
    const matches = (sign: Sign) => word(sign.sign).toLowerCase();
    const starts = signs.filter((sign) => matches(sign).startsWith(text));
    const inside = signs.filter((sign) => !matches(sign).startsWith(text) && matches(sign).includes(text));
    return starts.concat(inside).slice(0, 40);
  });
</script>

<section class="card">
  <h1>Öva ett tecken eller en mening</h1>
  <input type="search" placeholder="Sök efter ett ord" autocomplete="off" bind:value={query} />
  <div class="matches">
    {#each found as sign (sign.sign)}
      <button class="secondary" onclick={() => onpick(sign)}>{word(sign.sign)} ({entry(sign.sign)})</button>
    {/each}
  </div>
</section>

<style>
  .matches {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
  }
</style>
