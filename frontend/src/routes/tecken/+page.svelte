<script lang="ts">
  import {
    fetchLexicon,
    fetchSearch,
    referenceUrl,
    searchBySign,
    word,
    type Lexicon,
    type SignWord,
  } from "$lib/api";
  import { add, load, save, type Progress } from "$lib/progress";
  import { Tracker } from "$lib/tracking";

  // Tecken (steps 9 and 10 of ROADMAP-takk.md): the one way vocabulary grows. A word, a theme or a
  // sign shown to the camera lists signs to tick, and what is ticked is what Nya ord teaches.
  const WAIT = 200; // milliseconds after the last keystroke, so a word is searched once and not per letter

  let lexicon = $state<Lexicon | null>(null);
  let query = $state("");
  let results: SignWord[] = $state([]);
  let progress: Progress = $state({});
  let note = $state("");
  let timer: ReturnType<typeof setTimeout>;

  // Searching by signing: the camera takes the place of the field, and its recording fills the same
  // list of rows. Nothing is spoken and nothing is scored — this is a lookup, not an attempt.
  let camera = $state(false);
  let video = $state<HTMLVideoElement>(); // bound when the camera replaces the field, not before
  let canvas = $state<HTMLCanvasElement>();
  let tracker = $state<Tracker | null>(null);
  let starting: Promise<unknown> | null = null;
  let status = $state("Laddar teckenmodellen...");
  let handedness: "left" | "right" = $state("right");
  let recording = $state(false);
  let searching = $state(false);

  $effect(() => {
    progress = load();
    fetchLexicon()
      .then((loaded) => (lexicon = loaded))
      .catch(() => (note = "Servern svarar inte. Starta den med uv run takk."));
  });

  // The camera opens the first time it is asked for and keeps running, so a second lookup is instant.
  $effect(() => {
    if (!camera || !lexicon || !video || !canvas) return;
    starting ??= Tracker.start(video, canvas, lexicon.edges, (fps) => (status = `Följer tecknen i ${fps.toFixed(0)} fps`))
      .then((started) => (tracker = started))
      .catch((error) => (status = `Ingen åtkomst till kameran: ${error.message}`));
  });

  const clips = $derived(new Map(lexicon?.signs.map((each) => [each.sign, each.references]) ?? []));
  const label = (each: SignWord) => each.word ?? word(each.sign);

  function search(text: string) {
    clearTimeout(timer);
    query = text;
    timer = setTimeout(async () => (results = text.trim() ? await fetchSearch(text) : []), WAIT);
  }

  function record() {
    if (!tracker) return;
    if (recording) return void finish();
    tracker.resume();
    tracker.startRecording();
    (recording = true), (note = "");
  }

  async function finish() {
    recording = false;
    const taken = await tracker?.stopRecording();
    if (!taken || taken.frames.length < 2) return void (note = "Inspelningen är tom.");
    searching = true;
    try {
      const found = await searchBySign(taken.frames, handedness, video!, lexicon!.fps);
      (results = found.words), (note = found.note), (query = "");
      if (found.words.length) camera = false; // the rows take over the screen, as they do after a search
    } catch {
      note = "Sökningen misslyckades. Teckna igen.";
    } finally {
      searching = false;
    }
  }

  function pick(each: SignWord, on: boolean) {
    // Unticking is only ever undoing the tick: a sign that has been practised keeps its box, so its
    // checkbox stays on and disabled rather than throwing away what is learned of it.
    if (on) {
      progress = add(progress, [each]);
    } else {
      const { [each.sign]: dropped, ...rest } = progress;
      progress = rest;
    }
    save(progress);
  }

  function pickAll() {
    progress = add(progress, results);
    save(progress);
  }
</script>

{#if camera}
  <div class="view">
    <!-- svelte-ignore a11y_media_has_caption -->
    <video bind:this={video} class:recording autoplay muted playsinline></video>
    <canvas bind:this={canvas}></canvas>
  </div>
  <p class="dim">{status}</p>
  <p class="row">
    <button disabled={!tracker || searching} onclick={record}>{recording ? "Stopp" : "Starta"}</button>
    <button class="secondary" onclick={() => (camera = false)}>Sök med ord</button>
    <label class="row dim">
      Jag tecknar med
      <input type="radio" name="handedness" value="right" bind:group={handedness} /> höger
      <input type="radio" name="handedness" value="left" bind:group={handedness} /> vänster hand
    </label>
  </p>
  {#if searching}<p class="dim">Söker...</p>{/if}
{:else}
  <input
    type="search"
    placeholder="Sök efter ord eller teman"
    value={query}
    oninput={(event) => search(event.currentTarget.value)}
  />
  <p class="dim">eller</p>
  <button class="secondary" onclick={() => (camera = true)}>Sök med tecken</button>
{/if}

{#if note}
  <p class="dim">{note}</p>
{/if}

{#if results.length}
  <button onclick={pickAll}>Lägg till alla</button>

  <ul>
    {#each results as each (each.sign)}
      <li>
        <label>
          <input
            type="checkbox"
            checked={!!progress[each.sign]}
            disabled={progress[each.sign]?.box > 0}
            onchange={(event) => pick(each, event.currentTarget.checked)}
          />
          {label(each)}
        </label>
        {#if clips.get(each.sign)?.length}
          <!-- svelte-ignore a11y_media_has_caption -->
          <video src={referenceUrl(clips.get(each.sign)![0])} autoplay loop muted playsinline></video>
        {/if}
      </li>
    {/each}
  </ul>
{/if}

<style>
  button {
    margin: 12px 0;
  }

  .view {
    position: relative;
    width: 100%;
    max-width: 560px;
    transform: scaleX(-1);
  }

  .view video {
    display: block;
    width: 100%;
    border-radius: 8px;
    background: #000;
  }

  .view video.recording {
    outline: 3px solid var(--bad);
  }

  canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 6px 0;
    border-bottom: 1px solid var(--line);
  }

  li video {
    width: 140px;
    border-radius: 4px;
    background: #000;
  }
</style>
