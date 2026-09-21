<script lang="ts">
  import { boxes, DAYS, KEY, load, type Progress, type Row } from "$lib/progress";

  let progress: Progress = $state({});
  const now = Date.now();

  $effect(() => {
    progress = load();
  });

  const rows = $derived(boxes(progress));
  const picked = $derived(rows.filter((row) => row.box === 0).length);
  const due = $derived(rows.filter((row) => row.box > 0 && row.due <= now).length);
  const perBox = $derived(DAYS.map((_, i) => rows.filter((row) => row.box === i + 1).length));

  function when(row: Row): string {
    if (row.due <= now) return "nu";
    const days = Math.ceil((row.due - now) / (24 * 60 * 60 * 1000));
    return days === 1 ? "i morgon" : `om ${days} dagar`;
  }

  function reset() {
    localStorage.removeItem(KEY);
    progress = {};
  }
</script>

<section class="card">
  <h1>Mina tecken</h1>
  {#if rows.length}
    <p>
      {rows.length - picked} tecken övade, {picked} valda som väntar, {due} att repetera nu.
      <a href="/träna/repetera">Sagan</a> skrivs av dem du övat, med tyngdpunkt på dem som sitter sämst.
    </p>
    <ul class="boxes">
      {#each perBox as count, i (i)}
        <li>Låda {i + 1}: {count} tecken, repeteras efter {DAYS[i]} dagar</li>
      {/each}
    </ul>
  {:else}
    <p>Inga tecken valda än. Välj några under <a href="/tecken">Tecken</a>.</p>
  {/if}
</section>

{#if rows.length}
  <section class="card">
    <table>
      <thead>
        <tr><th>Tecken</th><th>Låda</th><th>Repeteras</th></tr>
      </thead>
      <tbody>
        {#each rows as row (row.sign)}
          <tr>
            <td>{row.word}</td>
            <td>{row.box}</td>
            <td class:due={row.box > 0 && row.due <= now}>{row.box === 0 ? "inte övat" : when(row)}</td>
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
