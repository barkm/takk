<script lang="ts">
  import { hand, setHand } from "$lib/hand";
  import { boxes, DAYS, KEY, load, type Progress } from "$lib/progress";

  // Tecken (step 14 of ROADMAP-takk.md): the learner's own signs. The numbers first, then how the
  // vocabulary is spread over the repetition schedule and how many signs have been met over time,
  // then the words themselves with the interval each one sits at. Everything is read off the store,
  // so no history is written for it. Drawn as inline SVG: one bar series and one line are not worth a
  // charting dependency.
  const PLOT = { width: 320, height: 140, pad: 22 }; // the viewBox both charts are drawn in

  let progress: Progress = $state({});
  let signs = $state<"left" | "right">("right"); // the hand the learner signs with, asked under Träna

  $effect(() => {
    (progress = load()), (signs = hand() ?? "right");
  });

  function swap() {
    signs = signs === "right" ? "left" : "right";
    setHand(signs);
  }

  const rows = $derived(boxes(progress));
  // one bar per state a sign can be in: picked but not taught, then a box per interval
  const labels = ["nya", ...DAYS.map((days) => (days === 1 ? "imorgon" : `${days} dagar`))];
  const bars = $derived(labels.map((label, box) => ({ label, count: rows.filter((row) => row.box === box).length })));
  const tallest = $derived(Math.max(1, ...bars.map((bar) => bar.count)));

  const practised = $derived(rows.filter((row) => row.box > 0).length);
  const due = $derived(rows.filter((row) => row.box > 0 && row.due <= Date.now()).length);

  /** What a word's row says about where it stands: the interval it has reached, or that it is new. */
  const interval = (box: number) =>
    box === 0 ? "ny" : DAYS[Math.min(box, DAYS.length) - 1] === 1 ? "1 dag" : `${DAYS[Math.min(box, DAYS.length) - 1]} dagar`;

  const when = (row: { box: number; due: number }) =>
    row.box === 0 ? "inte övat" : row.due <= Date.now() ? "nu" : new Date(row.due).toLocaleDateString("sv-SE", { day: "numeric", month: "short" });

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
  <section class="numbers">
    <p class="figure"><span>{rows.length}</span> valda tecken</p>
    <p class="figure"><span>{practised}</span> övade</p>
    <p class="figure"><span>{due}</span> att repetera nu</p>
  </section>

  <section class="charts">
    <figure class="card">
      <h2>Per repetitionsintervall</h2>
      <svg viewBox="0 0 {PLOT.width} {PLOT.height}" role="img" aria-label="Tecken per repetitionsintervall">
        {#each bars as bar, index (bar.label)}
          {@const slot = (PLOT.width - PLOT.pad * 2) / bars.length}
          {@const height = (bar.count / tallest) * (PLOT.height - PLOT.pad * 2)}
          <rect
            x={PLOT.pad + index * slot + 2}
            y={PLOT.height - PLOT.pad - height}
            width={slot - 4}
            height={bar.count ? Math.max(height, 2) : 0}
            rx="3"
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
    </figure>

    {#if line}
      <figure class="card">
        <h2>Tecken över tid</h2>
        <svg viewBox="0 0 {PLOT.width} {PLOT.height}" role="img" aria-label="Tecken över tid">
          <polyline class="over-time" points={line} />
          <line
            class="axis"
            x1={PLOT.pad}
            y1={PLOT.height - PLOT.pad}
            x2={PLOT.width - PLOT.pad}
            y2={PLOT.height - PLOT.pad}
          />
          <text class="value end" x={PLOT.width - PLOT.pad} y={PLOT.pad - 8}>{met.at(-1)?.count}</text>
          <text class="tick start" x={PLOT.pad} y={PLOT.height - PLOT.pad + 12}>{began}</text>
          <text class="tick end" x={PLOT.width - PLOT.pad} y={PLOT.height - PLOT.pad + 12}>idag</text>
        </svg>
      </figure>
    {/if}
  </section>

  <!-- the words themselves, soonest due first, which is the order `boxes` gives them in -->
  <ul class="words">
    {#each rows as row (row.sign)}
      <li>
        <span class="word">{row.word}</span>
        <span class="dim">{interval(row.box)}</span>
        <span class="dim due" class:now={row.box > 0 && row.due <= Date.now()}>{when(row)}</span>
      </li>
    {/each}
  </ul>

  <section class="settings">
    <p class="dim">Du tecknar med {signs === "right" ? "höger" : "vänster"} hand.</p>
    <div class="row">
      <button class="secondary" onclick={swap}>Byt hand</button>
      <button class="secondary" onclick={reset}>Nollställ</button>
    </div>
  </section>
{:else}
  <section class="card empty">
    <h1>Inga tecken än</h1>
    <p class="dim">Välj tecken under <a href="/sök">Sök</a>, så övas de här.</p>
  </section>
{/if}

<style>
  .numbers {
    display: flex;
    flex-wrap: wrap;
    gap: 32px;
    margin-bottom: 28px;
  }

  .figure {
    color: var(--dim);
    font-size: 14px;
  }

  .figure span {
    display: block;
    color: var(--text);
    font-size: 40px;
    font-weight: 650;
    letter-spacing: -0.02em;
    line-height: 1.1;
  }

  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
  }

  figure {
    margin: 0;
    gap: 12px;
  }

  svg {
    display: block;
    width: 100%;
  }

  .axis {
    stroke: var(--line);
    stroke-width: 1;
  }

  .over-time {
    fill: none;
    stroke: var(--accent);
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

  .words {
    list-style: none;
    margin: 28px 0;
    padding: 0;
    border-top: 1px solid var(--line);
  }

  .words li {
    display: grid;
    grid-template-columns: 1fr auto 90px;
    gap: 12px;
    align-items: baseline;
    padding: 12px 4px;
    border-bottom: 1px solid var(--line);
  }

  .word {
    font-weight: 500;
  }

  .due {
    text-align: right;
  }

  .due.now {
    color: var(--accent);
  }

  .settings {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .empty {
    max-width: 420px;
  }
</style>
