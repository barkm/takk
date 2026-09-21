<script lang="ts">
  import { fetchPacks, type Pack } from "$lib/api";
  import { boxes, DAYS, load, type Progress, type Row } from "$lib/progress";

  let packs: Pack[] = $state([]);
  let progress: Progress = $state({});
  let note = $state("");
  const now = Date.now();

  $effect(() => {
    progress = load();
    fetchPacks()
      .then((loaded) => (packs = loaded))
      .catch(() => (note = "Servern svarar inte, så tecknen visas med lexikonets egna namn."));
  });

  const rows = $derived(boxes(progress, packs));
  const due = $derived(rows.filter((row) => row.due <= now).length);
  const perBox = $derived(DAYS.map((_, i) => rows.filter((row) => row.box === i + 1).length));

  function when(row: Row): string {
    if (row.due <= now) return "nu";
    const days = Math.ceil((row.due - now) / (24 * 60 * 60 * 1000));
    return days === 1 ? "i morgon" : `om ${days} dagar`;
  }

  function reset() {
    localStorage.removeItem("takk.progress");
    progress = {};
  }
</script>

<section class="card">
  <h1>Mina tecken</h1>
  {#if rows.length}
    <p>
      {rows.length} tecken övade, {due} att repetera nu. <a href="/träna/repetera">Sagan</a> skrivs av dem, med
      tyngdpunkt på dem som sitter sämst.
    </p>
    <ul class="boxes">
      {#each perBox as count, i (i)}
        <li>Låda {i + 1}: {count} tecken, repeteras efter {DAYS[i]} dagar</li>
      {/each}
    </ul>
  {:else}
    <p>Inga tecken övade än. Börja med <a href="/träna/nya">nya ord</a>.</p>
  {/if}
  {#if note}<p class="dim">{note}</p>{/if}
</section>

{#if rows.length}
  <section class="card">
    <table>
      <thead>
        <tr><th>Tecken</th><th>Låda</th><th>Repeteras</th><th>Finns i</th></tr>
      </thead>
      <tbody>
        {#each rows as row (row.sign)}
          <tr>
            <td>{row.word}</td>
            <td>{row.box}</td>
            <td class:due={row.due <= now}>{when(row)}</td>
            <td class="dim">{row.packs.join(", ")}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </section>

  <section class="card">
    <details>
      <summary>Börja om</summary>
      <p class="dim">Nollställer alla lådor. Går inte att ångra, och inget sparas på servern.</p>
      <button class="secondary" onclick={reset}>Nollställ mina tecken</button>
    </details>
  </section>
{/if}

<style>
  .boxes {
    margin: 8px 0;
    padding-left: 20px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th {
    text-align: left;
  }

  th,
  td {
    padding: 4px 8px 4px 0;
    border-bottom: 1px solid #0002;
  }

  .due {
    font-weight: 600;
  }
</style>
