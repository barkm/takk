<script lang="ts">
  import { base } from "$app/paths";
  import { useCamera } from "$lib/camera.svelte";
  import { hand, setHand } from "$lib/hand";
  import { boxes, DAYS, KEY, load, remove, save, type Progress } from "$lib/progress";

  // Tecken (step 14 of ROADMAP-takk.md): the learner's own signs. The numbers first, then how the
  // vocabulary is spread over the repetition schedule and how many signs have been met over time,
  // then the signs themselves as a board: a card per word with its lexicon clip, grouped under the
  // step of the schedule it has reached (user, 2026-09-24). The board says what the bar chart says,
  // with the signs in it, which a list of words alone cannot show. Everything is read off the store,
  // so no history is written for it. The charts are inline SVG: one bar series and one line are not
  // worth a charting dependency.
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
          <span class="dots" aria-hidden="true">
            {#each DAYS as _, step (step)}
              <i class:on={step < row.box}></i>
            {/each}
          </span>
          {#if overdue(row)}<span class="now">nu</span>{/if}
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
    <p class="dim">Välj tecken under <a href="{base}/sök">Sök</a>, så övas de här.</p>
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
    border: 1px solid var(--line);
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

  /* how far up the schedule the word has come, one dot per box */
  .dots {
    display: flex;
    gap: 3px;
    padding-right: 12px;
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

  /* on the clip, so a card that is due is no taller than one that is not */
  .now {
    position: absolute;
    top: 6px;
    left: 6px;
    padding: 2px 8px;
    border-radius: 8px;
    background: var(--accent);
    color: #0b0b0d;
    font-size: 13px;
    font-weight: 600;
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
    color: var(--text);
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
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .empty {
    max-width: 420px;
  }
</style>
