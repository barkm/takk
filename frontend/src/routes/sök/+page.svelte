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
  import { framing, FRAMING } from "$lib/framing";
  import { draw, type Frame } from "$lib/landmarks";
  import { add, load, save, type Progress } from "$lib/progress";
  import { Tracker } from "$lib/tracking";

  // Sök (steps 9 and 10 of ROADMAP-takk.md): the one way vocabulary grows. A word, a theme or a
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
  // Once opened, the camera stays in the page and is only hidden: the tracker holds the video element
  // it was started with and draws into its canvas, so unmounting them leaves a black picture behind.
  let opened = $state(false);
  let video = $state<HTMLVideoElement>(); // bound when the camera replaces the field, not before
  let canvas = $state<HTMLCanvasElement>();
  let tracker = $state<Tracker | null>(null);
  let starting: Promise<unknown> | null = null;
  let status = $state("Laddar teckenmodellen...");
  const LOOK = 200; // ms between readings of how the signer sits; faster than that only flickers
  let handedness: "left" | "right" = $state("right");
  let recording = $state(false);
  let searching = $state(false);
  let fit = $state(""); // what to fix about the framing, or FRAMING.ok
  // What the rows were searched with, kept as the field keeps the word that found them. Only the
  // landmarks are kept, never the camera's picture, so the replay is the skeleton that was sent.
  let signed: Frame[] = $state([]);
  let replay = $state<HTMLCanvasElement>();

  $effect(() => {
    progress = load();
    fetchLexicon()
      .then((loaded) => (lexicon = loaded))
      .catch(() => (note = "Servern svarar inte. Starta den med uv run takk."));
  });

  // The camera opens the first time it is asked for and keeps running, so a second lookup is instant.
  $effect(() => {
    if (!opened || !lexicon || !video || !canvas) return;
    starting ??= Tracker.start(video, canvas, lexicon.edges)
      .then((started) => ((tracker = started), (status = "")))
      .catch((error) => (status = `Ingen åtkomst till kameran: ${error.message}`));
  });

  // How the signer sits, read off the frame being tracked right now, so it is fixed before a
  // recording is spent on it. While recording it also asks for the hands.
  $effect(() => {
    if (!camera || !tracker) return;
    const looking = setInterval(() => {
      fit = tracker?.latest ? framing(tracker.latest, recording) : FRAMING.none;
    }, LOOK);
    return () => clearInterval(looking);
  });

  const clips = $derived(new Map(lexicon?.signs.map((each) => [each.sign, each.references]) ?? []));
  const label = (each: SignWord) => each.word ?? word(each.sign);

  // The recording plays on a loop beside its results, a frame at the rate the landmarks were sent at.
  $effect(() => {
    const canvas = replay;
    if (!canvas || !signed.length || !lexicon || !video) return;
    // the canvas keeps the camera's own proportions, since the landmarks are normalised to its frame
    const size = { width: video.videoWidth, height: video.videoHeight };
    let at = 0;
    const shown = setInterval(() => {
      draw(canvas, size, signed[at].landmarks, lexicon!.edges);
      at = (at + 1) % signed.length;
    }, 1000 / lexicon.fps);
    return () => clearInterval(shown);
  });

  function search(text: string) {
    clearTimeout(timer);
    (query = text), (signed = []); // a word search replaces what the rows were found with
    timer = setTimeout(async () => (results = text.trim() ? await fetchSearch(text) : []), WAIT);
  }

  function record() {
    if (!tracker) return;
    if (recording) return void finish();
    tracker.resume();
    tracker.startRecording();
    (recording = true), (note = ""), (signed = []); // the live picture, until this recording replaces it
  }

  async function finish() {
    recording = false;
    const taken = await tracker?.stopRecording();
    if (!taken || taken.frames.length < 2) return void (note = "Inspelningen är tom.");
    searching = true;
    try {
      const found = await searchBySign(taken.frames, handedness, video!, lexicon!.fps);
      (results = found.words), (note = found.note), (query = "");
      signed = found.words.length ? taken.frames : []; // the camera stays, with the sign that found the rows in it
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

{#if opened}
  <div class="view" class:away={!camera}>
    <!-- svelte-ignore a11y_media_has_caption -->
    <video bind:this={video} class:recording autoplay muted playsinline></video>
    <canvas bind:this={canvas}></canvas>
    <!-- the sign the rows were found by, in the camera's own place, until the next recording -->
    <canvas bind:this={replay} class="replay" class:away={!signed.length}></canvas>
  </div>
{/if}

{#if camera}
  <p class="dim">{status || (signed.length ? "" : fit)}</p>
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
  <button class="secondary" onclick={() => ((camera = true), (opened = true))}>Sök med tecken</button>
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

  .away {
    display: none; /* hidden rather than removed, so the tracker keeps the element it started on */
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

  .replay {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    background: #000; /* over the live picture, so what is shown is what was searched with */
    border-radius: 8px;
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
