<script lang="ts">
  import {
    fetchSearch,
    referenceUrl,
    searchBySign,
    word,
    type SignWord,
  } from "$lib/api";
  import CameraView from "$lib/Camera.svelte";
  import { useCamera } from "$lib/camera.svelte";
  import { hand } from "$lib/hand";
  import { draw, type Frame } from "$lib/landmarks";
  import { add, load, remove, save, type Progress } from "$lib/progress";

  // Sök (steps 9 and 10 of ROADMAP-takk.md): the one way vocabulary grows. A word, a theme or a
  // sign shown to the camera lists signs to pick, and what is picked is what Ord teaches.
  const WAIT = 200; // milliseconds after the last keystroke, so a word is searched once and not per letter

  let query = $state("");
  let results: SignWord[] = $state([]);
  let progress: Progress = $state({});
  let note = $state("");
  let timer: ReturnType<typeof setTimeout>;

  // Searching by signing: the recording fills the same grid the field does. Nothing is spoken and
  // nothing is scored — this is a lookup, not an attempt. The camera is only mounted while this is
  // the way being searched (user, 2026-09-23), so a page of results costs nothing.
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

  // The replay goes in the camera's own place, and leaves it when the search by sign is left.
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

  function pick(each: SignWord) {
    // A picked sign can always be unpicked (user, 2026-09-24), practised or not: the tile is the one
    // switch for whether the word is in the vocabulary, and unpicking drops what was learned of it.
    progress = picked(each) ? remove(progress, each.sign) : add(progress, [each]);
    save(progress);
  }

  function pickAll() {
    progress = add(progress, results);
    save(progress);
  }

  const picked = (each: SignWord) => !!progress[each.sign];
  const missing = $derived(results.filter((each) => !progress[each.sign]).length);
</script>

<!-- the sign the rows were found by, in the camera's own place, until the next recording -->
{#snippet replayed()}
  <canvas bind:this={replay} class="replay" class:away={!signed.length}></canvas>
{/snippet}

<header class="search">
  <input
    type="search"
    placeholder="Sök efter ord eller teman"
    value={query}
    oninput={(event) => search(event.currentTarget.value)}
  />
  <button class="secondary" onclick={() => (bySign = !bySign)}>
    {bySign ? "Sök med ord" : "Sök med tecken"}
  </button>
</header>

{#if bySign}
  <section class="signing">
    <div class="picture"><CameraView {camera} /></div>
    <div class="row">
      <button disabled={!camera.tracker || searching} onclick={record}>
        {camera.recording ? "Stopp" : "Spela in tecken"}
      </button>
      {#if searching}<span class="dim">Söker ...</span>{/if}
    </div>
  </section>
{/if}

{#if note}
  <p class="dim note">{note}</p>
{/if}

{#if !results.length && !bySign && !query}
  <p class="dim note">Sök på ett ord eller ett tema, till exempel "mat" eller "känslor".</p>
{/if}

{#if results.length}
  <div class="found">
    <span class="dim">{results.length} tecken</span>
    <button class="secondary" onclick={pickAll} disabled={!missing}>Lägg till alla</button>
  </div>

  <ul class="grid">
    {#each results as each (each.sign)}
      {@const clip = clips.get(each.sign)?.[0]}
      <li>
        <button
          class="tile"
          class:on={picked(each)}
          aria-pressed={picked(each)}
          onclick={() => pick(each)}
        >
          {#if clip}
            <!-- svelte-ignore a11y_media_has_caption -->
            <video src={referenceUrl(clip)} autoplay loop muted playsinline></video>
          {:else}
            <span class="noclip dim">inget klipp</span>
          {/if}
          <span class="name">
            {label(each)}
            <span class="mark">{picked(each) ? "Vald" : "Lägg till"}</span>
          </span>
        </button>
      </li>
    {/each}
  </ul>
{/if}

<style>
  .search {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .search input {
    flex: 1;
  }

  .search button {
    white-space: nowrap;
  }

  .signing {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 20px;
    max-width: 420px;
  }

  .picture {
    position: relative;
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
  }

  .note {
    margin-top: 16px;
  }

  .found {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin: 28px 0 12px;
  }

  /* The results are their clips: a word is recognised by the sign, not by the row it sits in. */
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 14px;
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .tile {
    display: block;
    width: 100%;
    padding: 0;
    overflow: hidden;
    text-align: left;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    color: var(--text);
    font-weight: 500;
  }

  .tile:hover {
    filter: none;
    border-color: var(--dim);
  }

  .tile.on {
    border-color: var(--accent);
  }

  /* the lexicon's own shape, held before the clip loads so the grid does not reflow under the cursor */
  .tile video,
  .noclip {
    display: block;
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
  }

  .noclip {
    display: grid;
    place-items: center;
    background: var(--raised);
    font-size: 13px;
  }

  .name {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 12px 14px;
  }

  .mark {
    font-size: 13px;
    font-weight: 500;
    color: var(--dim);
  }

  .tile.on .mark {
    color: var(--accent);
  }
</style>
