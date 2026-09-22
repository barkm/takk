<script lang="ts">
  import {
    fetchSearch,
    referenceUrl,
    searchBySign,
    word,
    type SignWord,
  } from "$lib/api";
  import { useCamera } from "$lib/camera.svelte";
  import { hand } from "$lib/hand";
  import { draw, type Frame } from "$lib/landmarks";
  import { add, load, save, type Progress } from "$lib/progress";

  // Sök (steps 9 and 10 of ROADMAP-takk.md): the one way vocabulary grows. A word, a theme or a
  // sign shown to the camera lists signs to tick, and what is ticked is what Ord teaches.
  const WAIT = 200; // milliseconds after the last keystroke, so a word is searched once and not per letter

  let query = $state("");
  let results: SignWord[] = $state([]);
  let progress: Progress = $state({});
  let note = $state("");
  let timer: ReturnType<typeof setTimeout>;

  // Searching by signing: the recording fills the same list of rows as the field does. Nothing is
  // spoken and nothing is scored — this is a lookup, not an attempt. The picture is the app's own,
  // above this page, so all that changes here is what is asked of it.
  const camera = useCamera();
  const lexicon = $derived(camera.lexicon);
  let bySign = $state(false);
  let searching = $state(false);
  // What the rows were searched with, kept as the field keeps the word that found them. Only the
  // landmarks are kept, never the camera's picture, so the replay is the skeleton that was sent.
  let signed: Frame[] = $state([]);
  let replay = $state<HTMLCanvasElement>();

  $effect(() => {
    progress = load();
  });

  // The replay goes in the app's picture, which is above this page, and leaves it when the page does.
  $effect(() => {
    camera.overlay = replayed;
    return () => (camera.overlay = null);
  });

  const clips = $derived(new Map(lexicon?.signs.map((each) => [each.sign, each.references]) ?? []));
  const label = (each: SignWord) => each.word ?? word(each.sign);

  // The recording plays on a loop beside its results, a frame at the rate the landmarks were sent at.
  $effect(() => {
    const canvas = replay;
    if (!canvas || !signed.length || !lexicon || !camera.video) return;
    // the canvas keeps the camera's own proportions, since the landmarks are normalised to its frame
    const size = { width: camera.video.videoWidth, height: camera.video.videoHeight };
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
    timer = setTimeout(async () => {
      results = text.trim() ? await fetchSearch(text) : [];
    }, WAIT);
  }

  function record() {
    if (!camera.tracker) return;
    if (camera.recording) return void finish();
    camera.tracker.resume();
    camera.tracker.startRecording();
    (camera.recording = true), (note = ""), (signed = []); // the live picture, until this replaces it
  }

  async function finish() {
    camera.recording = false;
    const taken = await camera.tracker?.stopRecording();
    if (!taken || taken.frames.length < 2) return void (note = "Inspelningen är tom.");
    searching = true;
    try {
      const found = await searchBySign(taken.frames, hand() ?? "right", camera.video!, lexicon!.fps);
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

<!-- the sign the rows were found by, in the camera's own place, until the next recording -->
{#snippet replayed()}
  <canvas bind:this={replay} class="replay" class:away={!signed.length}></canvas>
{/snippet}

{#if bySign}
  <p class="row">
    <button disabled={!camera.tracker || searching} onclick={record}>
      {camera.recording ? "Stopp" : "Starta"}
    </button>
    <button class="secondary" onclick={() => (bySign = false)}>Sök med ord</button>
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
  <button class="secondary" onclick={() => (bySign = true)}>Sök med tecken</button>
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

  .replay {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    background: #000; /* over the live picture, so what is shown is what was searched with */
    border-radius: 12px;
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
    padding: 10px 0;
    text-align: left;
    border-bottom: 1px solid var(--line);
  }

  /* the lexicon's own shape, held before the clip loads so the row does not grow under the cursor */
  li video {
    width: 140px;
    aspect-ratio: 4 / 3;
    object-fit: cover;
    border-radius: 4px;
    background: #000;
  }
</style>
