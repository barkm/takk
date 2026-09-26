<script lang="ts">
  import { afterNavigate } from "$app/navigation";
  import { useCamera } from "$lib/camera.svelte";
  import { known, load, toLearn } from "$lib/progress";
  import Meningar from "./Meningar.svelte";
  import Ord from "./Ord.svelte";

  // Träna is one page (user, 2026-09-26): the start card of both modes, a button for each, and the
  // mode itself once one is pressed. With nothing left to choose before a pass, the two start cards
  // were a heading, a line and "Börja" each, behind a switch of their own. A mode that ends hands the
  // page back to the start card, and so does pressing "Träna" in the menu while one is running — the
  // way out of a pass, since there is no switch to leave it by any more.
  const camera = useCamera();
  let mode = $state<"ord" | "meningar" | null>(null);
  let note = $state(""); // why Meningar could not start, under its button
  let progress = $state(load());

  afterNavigate(() => (mode = null));

  function done(told = "") {
    (note = told), (progress = load()), (mode = null);
  }

  const waiting = $derived(toLearn(progress)); // what Ord would teach: missed words, then new ones
  const pool = $derived(known(progress, Infinity, () => 0)); // what Meningar can be written over
</script>

{#if !camera.lexicon}
  <p class="dim">Laddar lexikonet ...</p>
{:else if mode === "ord"}
  <Ord ondone={done} />
{:else if mode === "meningar"}
  <Meningar ondone={done} />
{:else}
  <!-- The two modes as two large tiles side by side (user, 2026-09-26), the whole tile the button. -->
  <div class="modes">
    <button class="mode" onclick={() => (mode = "ord")} disabled={!waiting.length}>
      <span class="text">
        <span class="name">Ord</span>
        <span class="what">Tecken du inte övat än, eller tecknade fel.</span>
      </span>
      {#if !waiting.length}<span class="why">Inga tecken att öva. Välj några under Sök.</span>{:else}<svg class="go" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>{/if}
    </button>
    <button class="mode" onclick={() => (mode = "meningar")} disabled={pool.length < 2}>
      <span class="text">
        <span class="name">Meningar</span>
        <span class="what">En text över orden du redan kan, en rad i taget.</span>
      </span>
      {#if pool.length < 2}<span class="why">Öva några ord först.</span>{:else}<svg class="go" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>{/if}
    </button>
  </div>
  {#if note}<p class="dim note">{note}</p>{/if}
{/if}

<style>
  /* The two modes as two bars, together exactly as tall as the camera beside them on a desktop
     window: the column is stretched to the camera's height, and the level strip under the picture
     (24 px and its 8 px margin) is left out so the last bar ends where the picture does. */
  .modes {
    flex: 1;
    display: grid;
    grid-template-rows: 1fr 1fr;
    gap: 16px;
    margin-bottom: 32px;
  }

  .mode {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    min-height: 88px;
    padding: 16px 28px;
    text-align: left;
    background: var(--surface);
    color: var(--text);
  }

  /* Grey bars with the accent only in the arrow (user, 2026-09-26): solid blue made them the loudest
     thing in the app. The arrow says a bar can be pressed; a bar that cannot shows why instead. */
  .mode:hover:not(:disabled) {
    background: var(--raised);
    filter: none;
  }

  .go {
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    fill: none;
    stroke: var(--accent);
    stroke-width: 2.5;
    stroke-linecap: round;
    stroke-linejoin: round;
    transition: transform 0.15s ease;
  }

  .mode:hover .go {
    transform: translateX(3px);
  }

  /* a bar that cannot be pressed keeps its grey, its text dimmed rather than faded, so it stays readable */
  .mode:disabled {
    opacity: 1;
    color: var(--dim);
  }

  .text {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .name {
    font-size: 24px;
    font-weight: 700;
  }

  .what {
    font-size: 14px;
    font-weight: 400;
    line-height: 1.4;
    color: var(--dim);
  }

  .why {
    flex-shrink: 0;
    max-width: 40%;
    font-size: 14px;
    font-weight: 600;
    text-align: right;
  }

  .note {
    margin-top: 16px;
  }
</style>
