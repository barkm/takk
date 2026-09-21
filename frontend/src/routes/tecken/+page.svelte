<script lang="ts">
  import { boxes, DAYS, KEY, load, type Progress } from "$lib/progress";

  // Tecken (step 14 of ROADMAP-takk.md): the learner's own signs, as two charts and a reset. The bars say how
  // the vocabulary is spread over the repetition schedule, one bar per state; the line says how many
  // signs the learner has met over time. Both are read off the store, so no history is written for
  // them. Drawn as inline SVG: one bar series and one line are not worth a charting dependency.
  const PLOT = { width: 320, height: 140, pad: 22 }; // the viewBox both charts are drawn in

  let progress: Progress = $state({});

  $effect(() => {
    progress = load();
  });

  const rows = $derived(boxes(progress));
  // one bar per state a sign can be in: picked but not taught, then a box per interval
  const labels = ["nya", ...DAYS.map((days) => (days === 1 ? "imorgon" : `${days} dagar`))];
  const bars = $derived(labels.map((label, box) => ({ label, count: rows.filter((row) => row.box === box).length })));
  const tallest = $derived(Math.max(1, ...bars.map((bar) => bar.count)));

  /** The signs met over time, counted cumulatively: one point per sign, at the day it was first
   * practised. A sign picked and not yet practised has no such day, so it has not been met. */
  const met = $derived.by(() => {
    const days = rows
      .map((row) => row.first)
      .filter((first): first is number => !!first)
      .sort((a, b) => a - b);
    return days.length ? [{ day: days[0], count: 0 }, ...days.map((day, index) => ({ day, count: index + 1 }))] : [];
  });

  /** The day the line starts on, written as the learner reads a date ("3 sep."). The axis runs from
   * there to now, and without the two ends written out a week and a year look the same. */
  const began = $derived(
    met.length ? new Date(met[0].day).toLocaleDateString("sv-SE", { day: "numeric", month: "short" }) : "",
  );

  const line = $derived.by(() => {
    if (met.length < 2) return "";
    const first = met[0].day;
    const span = Math.max(1, Date.now() - first);
    const high = met.at(-1)!.count;
    return met
      .map(({ day, count }) => {
        const x = PLOT.pad + ((day - first) / span) * (PLOT.width - PLOT.pad * 2);
        const y = PLOT.height - PLOT.pad - (count / high) * (PLOT.height - PLOT.pad * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  });

  function reset() {
    localStorage.removeItem(KEY);
    progress = {};
  }
</script>

{#if rows.length}
  <svg viewBox="0 0 {PLOT.width} {PLOT.height}" role="img" aria-label="Tecken per repetitionsintervall">
    {#each bars as bar, index (bar.label)}
      {@const slot = (PLOT.width - PLOT.pad * 2) / bars.length}
      {@const height = (bar.count / tallest) * (PLOT.height - PLOT.pad * 2)}
      <rect
        x={PLOT.pad + index * slot + 2}
        y={PLOT.height - PLOT.pad - height}
        width={slot - 4}
        height={bar.count ? Math.max(height, 2) : 0}
        rx="4"
        fill="var(--accent)"
      />
      <text class="value" x={PLOT.pad + index * slot + slot / 2} y={PLOT.height - PLOT.pad - height - 5}>
        {bar.count || ""}
      </text>
      <text class="tick" x={PLOT.pad + index * slot + slot / 2} y={PLOT.height - PLOT.pad + 12}>{bar.label}</text>
    {/each}
    <line
      class="axis"
      x1={PLOT.pad}
      y1={PLOT.height - PLOT.pad}
      x2={PLOT.width - PLOT.pad}
      y2={PLOT.height - PLOT.pad}
    />
  </svg>

  {#if line}
    <svg viewBox="0 0 {PLOT.width} {PLOT.height}" role="img" aria-label="Tecken över tid">
      <polyline class="over-time" points={line} />
      <line
        class="axis"
        x1={PLOT.pad}
        y1={PLOT.height - PLOT.pad}
        x2={PLOT.width - PLOT.pad}
        y2={PLOT.height - PLOT.pad}
      />
      <text class="tick start" x={PLOT.pad} y={PLOT.pad - 8}>tecken</text>
      <text class="value end" x={PLOT.width - PLOT.pad} y={PLOT.pad - 8}>{met.at(-1)?.count}</text>
      <text class="tick start" x={PLOT.pad} y={PLOT.height - PLOT.pad + 12}>{began}</text>
      <text class="tick end" x={PLOT.width - PLOT.pad} y={PLOT.height - PLOT.pad + 12}>idag</text>
    </svg>
  {/if}

  <button class="secondary" onclick={reset}>Nollställ</button>
{:else}
  <p class="dim">Inga tecken valda än. Välj några under <a href="/sök">Sök</a>.</p>
{/if}

<style>
  svg {
    display: block;
    width: 100%;
    max-width: 560px;
    margin-bottom: 16px;
  }

  .axis {
    stroke: var(--line);
    stroke-width: 1;
  }

  .over-time {
    fill: none;
    stroke: var(--ok);
    stroke-width: 2;
    stroke-linejoin: round;
  }

  .tick {
    fill: var(--dim);
    font-size: 8px;
    text-anchor: middle;
  }

  .value {
    fill: var(--text);
    font-size: 9px;
    text-anchor: middle;
  }

  .start {
    text-anchor: start;
  }

  .end {
    text-anchor: end;
  }
</style>
