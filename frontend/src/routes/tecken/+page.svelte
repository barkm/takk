<script lang="ts">
  import { base } from "$app/paths";
  import { useCamera } from "$lib/camera.svelte";
  import { hand, setHand } from "$lib/hand";
  import { boxes, DAYS, KEY, load, remove, save, type Progress } from "$lib/progress";
  import { curveStepAfter } from "d3-shape";
  import { AreaChart, BarChart } from "layerchart";
  import "layerchart/core.css";

  // Tecken (step 14 of ROADMAP-takk.md): the learner's own signs. The numbers first, then how the
  // vocabulary is spread over the repetition schedule and how many signs have been met over time,
  // then the signs themselves as a board: a card per word with its lexicon clip, grouped under the
  // step of the schedule it has reached (user, 2026-09-24). The board says what the bar chart says,
  // with the signs in it, which a list of words alone cannot show. Everything is read off the store,
  // so no history is written for it. The charts are LayerChart's, which draws better axes and
  // hovers than hand-made SVG did (user, 2026-09-26).

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

  // The board: one group per step of the schedule, in the order a word climbs them, and the words of
  // each group soonest due first, which is the order `boxes` gives them in. A step with no words is
  // left out rather than standing empty — the bar chart above is where the zeroes are read.
  const groups = $derived(
    labels
      .map((label, box) => ({ label, box, words: rows.filter((row) => row.box === box) }))
      .filter((group) => group.words.length),
  );

  // The clips are the lexicon's, which the app loads once into the camera context; this page shows
  // them but opens no camera.
  const camera = useCamera();
  const clips = $derived(new Map(camera.lexicon?.signs.map((each) => [each.sign, each.references]) ?? []));

  const overdue = (row: { box: number; due: number }) => row.box > 0 && row.due <= Date.now();

  /** The signs met over time, counted cumulatively: one point per sign, at the day it was first
   * practised, and a last one today so the axis runs to now. A sign picked and not yet practised has
   * no such day, so it has not been met. */
  const met = $derived.by(() => {
    const days = rows
      .map((row) => row.first)
      .filter((first): first is number => !!first)
      .sort((a, b) => a - b);
    if (!days.length) return [];
    const points = [{ day: new Date(days[0]), count: 0 }, ...days.map((day, index) => ({ day: new Date(day), count: index + 1 }))];
    return [...points, { day: new Date(), count: days.length }];
  });

  // One padding for both charts, so their baselines sit on one line; the room on the right is for
  // the count written at the end of the line.
  const padding = { top: 16, right: 20, bottom: 24, left: 4 };

  /** A date as the learner reads it ("3 sep.", or "idag"). */
  const date = (day: Date) =>
    day.toDateString() === new Date().toDateString() ? "idag" : day.toLocaleDateString("sv-SE", { day: "numeric", month: "short" });

  // The axis is written at its two ends, or once when both fall on the same day: every sign met today
  // would otherwise read "idag – idag".
  const ends = $derived(
    met.length && date(met[0].day) !== date(met[met.length - 1].day) ? [met[0].day, met[met.length - 1].day] : met.slice(-1).map((point) => point.day),
  );

  // The list is where the vocabulary is pruned: a word is dropped here as it is unpicked in Sök, and
  // what was learned of it goes with it (user, 2026-09-24).
  function drop(sign: string) {
    progress = remove(progress, sign);
    save(progress);
  }

  function reset() {
    localStorage.removeItem(KEY);
    progress = {};
  }
</script>

{#if rows.length}
  <section class="numbers">
    <p class="figure"><span>{rows.length}</span> valda tecken</p>
    <p class="figure"><span>{practised}</span> övade</p>
    <p class="figure" class:due={due > 0}><span>{due}</span> att repetera nu</p>
  </section>

  <section class="charts">
    <figure class="tile">
      <figcaption>Per repetitionsintervall</figcaption>
      <div class="plot">
        <BarChart
          data={bars}
          x="label"
          series={[{ key: "count", label: "tecken", color: "var(--accent)" }]}
          axis="x"
          grid={false}
          bandPadding={0.65}
          {padding}
          labels={{ format: (count: number) => (count ? String(count) : "") }}
          props={{ bars: { strokeWidth: 0, radius: 4 }, xAxis: { tickMarks: false } }}
        />
      </div>
    </figure>

    {#if met.length}
      <figure class="tile">
        <figcaption>Tecken över tid</figcaption>
        <div class="plot line">
          <!-- A step, since the count only changes on the days signs are met, with a wash under it and
               the count of today written at its end in place of a count axis (user, 2026-09-26). -->
          <AreaChart
            data={met}
            x="day"
            series={[{ key: "count", label: "tecken", color: "var(--accent)" }]}
            axis="x"
            grid={false}
            {padding}
            points={{ data: met.slice(-1), r: 4 }}
            labels={{ data: met.slice(-1), placement: "outside", offset: 8, format: (count: number) => String(count) }}
            props={{
              area: { curve: curveStepAfter, fillOpacity: 0.25, line: { strokeWidth: 2.5 } },
              xAxis: { format: date, ticks: ends, tickMarks: false },
              tooltip: { header: { format: date } },
            }}
          />
        </div>
      </figure>
    {/if}
  </section>

  <!-- the signs themselves, grouped by the step of the schedule they have reached -->
  {#each groups as group (group.box)}
    <h2 class="group">{group.label} <span class="dim">{group.words.length}</span></h2>
    <ul class="board">
      {#each group.words as row (row.sign)}
        {@const clip = clips.get(row.sign)?.[0]}
        <li class:due={overdue(row)}>
          {#if clip}
            <!-- svelte-ignore a11y_media_has_caption -->
            <!-- still until it is pointed at, since a page of vocabulary is a page of moving pictures -->
            <!-- the fragment seeks into the clip, so the card shows the sign being made rather than
                 the signer still at rest; 0.6 s is short enough for all but the shortest clips -->
            <video
              src="{clip}#t=0.6"
              muted
              loop
              playsinline
              preload="metadata"
              onpointerenter={(event) => event.currentTarget.play()}
              onpointerleave={(event) => event.currentTarget.pause()}
            ></video>
          {/if}
          <span class="word">{row.word}</span>
          <span class="side">
            {#if overdue(row)}<span class="now">nu</span>{/if}
            <span class="dots" aria-hidden="true">
              {#each DAYS as _, step (step)}
                <i class:on={step < row.box}></i>
              {/each}
            </span>
          </span>
          <!-- the cross is drawn rather than written: a glyph sits on the font's own axis, which is
               not the middle of the button it is centred in -->
          <button class="drop" aria-label="Ta bort {row.word}" onclick={() => drop(row.sign)}>
            <svg viewBox="0 0 24 24" width="11" height="11" aria-hidden="true">
              <path d="M4 4 20 20M20 4 4 20" stroke="currentColor" stroke-width="3" stroke-linecap="round" />
            </svg>
          </button>
        </li>
      {/each}
    </ul>
  {/each}

  <!-- what is set rather than what is seen, as one quiet line at the foot of the page -->
  <footer class="settings dim">
    Du tecknar med {signs === "right" ? "höger" : "vänster"} hand ·
    <button class="link" onclick={swap}>Byt hand</button> ·
    <button class="link" onclick={reset}>Nollställ</button>
  </footer>
{:else}
  <section class="card empty">
    <h1>Inga tecken än</h1>
    <p class="dim">Välj tecken under <a href="{base}/sök">Sök</a>, så övas de här.</p>
  </section>
{/if}

<style>
  /* The three numbers as tiles, grey like the bars under Träna (user, 2026-09-26). */
  .numbers {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 16px; /* the gap between tiles, so numbers and charts read as one block */
  }

  .figure {
    padding: 20px 24px;
    border-radius: var(--radius);
    background: var(--surface);
    color: var(--dim);
    font-size: 14px;
  }

  /* what is due is the one number that asks for something, so it is the one in the accent */
  .figure.due span {
    color: var(--accent);
  }

  .figure span {
    display: block;
    color: var(--text);
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.1;
  }

  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
  }

  /* Each chart on a tile like the numbers above it (user, 2026-09-26), its title as small and dim as
     theirs, so the data is the loud part. The grid shares its gap with the numbers', so the outer
     edges line up. */
  .tile {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin: 0;
    padding: 20px 24px 12px;
    border-radius: var(--radius);
    background: var(--surface);
  }

  figcaption {
    color: var(--dim);
    font-size: 14px;
  }

  /* LayerChart fills its container, so the container sets the height. Its axis text recedes in the
     page's dim ink, so the marks are what is read. */
  .plot {
    height: 160px;
  }

  .plot :global(.lc-axis-tick-label) {
    font-size: 12px;
    fill: var(--dim);
  }

  /* the counts over the bars and at the end of the line, which are what the charts are read for */
  .plot :global(.lc-labels-text) {
    font-size: 13px;
    font-weight: 600;
    fill: var(--text);
  }

  /* the line's two dates sit under its two ends rather than hanging past them */
  .line :global(.lc-axis-tick-group:first-child .lc-axis-tick-label) {
    text-anchor: start;
  }

  .line :global(.lc-axis-tick-group:last-child:not(:first-child) .lc-axis-tick-label) {
    text-anchor: end;
  }

  /* One heading per step of the schedule, written as the chart's own tick is written. */
  .group {
    margin: 28px 0 12px;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.01em;
  }

  .group::first-letter {
    text-transform: uppercase;
  }

  .group .dim {
    margin-left: 6px;
    font-weight: 400;
  }

  /* The board: a card per word, small enough that a vocabulary is seen at once. */
  .board {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px;
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .board li {
    position: relative;
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 4px 8px;
    padding-bottom: 10px;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid transparent; /* grey like every tile; only a card that is due is outlined */
    border-radius: var(--radius);
  }

  /* a word the schedule asks for now, marked on the card itself rather than only in the numbers */
  .board li.due {
    border-color: var(--accent);
  }

  /* the lexicon's own shape, held before the clip loads so the board does not reflow as it fills */
  .board video {
    grid-column: 1 / -1;
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
    margin-bottom: 8px;
  }

  .word {
    padding-left: 12px;
    font-weight: 500;
  }

  .side {
    display: flex;
    align-items: center;
    gap: 6px;
    padding-right: 12px;
  }

  /* how far up the schedule the word has come, one dot per box */
  .dots {
    display: flex;
    gap: 3px;
  }

  .dots i {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--line);
  }

  .dots i.on {
    background: var(--accent);
  }

  /* beside the dots rather than on the clip, where it covered the word of a card without one */
  .now {
    padding: 0 6px;
    border-radius: 6px;
    background: var(--accent);
    color: var(--on-accent);
    font-size: 12px;
    font-weight: 600;
    line-height: 18px;
  }

  /* over the clip, in the corner of the card: a square with the drawn cross centred in it */
  .drop {
    position: absolute;
    top: 6px;
    right: 6px;
    display: grid;
    place-items: center;
    width: 26px;
    height: 26px;
    padding: 0;
    background: #000000a6;
    border-radius: 8px;
    color: #fff;
  }

  .drop svg {
    display: block;
    fill: none;
  }

  .drop:hover {
    color: var(--bad);
    filter: none;
  }

  /* Where there is a cursor it stays out of the way until the card is under it, since removing a word
     is not what the board is for. Where there is none it is always there, or it could not be hit. */
  @media (hover: hover) {
    .drop {
      opacity: 0;
    }

    .board li:hover .drop,
    .drop:focus-visible {
      opacity: 1;
    }
  }

  .settings {
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid var(--line);
  }

  /* the settings are links in the line, not buttons to press */
  .link {
    padding: 0;
    background: none;
    color: var(--text);
    font-weight: 500;
    text-decoration: underline;
    text-decoration-color: var(--line);
    text-underline-offset: 3px;
  }

  .link:hover:not(:disabled) {
    filter: none;
    text-decoration-color: var(--accent);
  }

  .empty {
    max-width: 420px;
  }
</style>
