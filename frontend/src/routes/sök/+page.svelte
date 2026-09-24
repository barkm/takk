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
  import { hands, see, watching } from "$lib/hands";
  import { add, load, remove, save, type Progress } from "$lib/progress";

  // Sök (steps 9 and 10 of ROADMAP-takk.md): the one way vocabulary grows. A word, a theme or a
  // sign shown to the camera lists signs to pick, and what is picked is what Ord teaches.
  const WAIT = 200; // milliseconds after the last keystroke, so a word is searched once and not per letter
  const LOOK = 50; // ms between readings of whether the hands are up, as the voice is read in Träna

  let query = $state("");
  let results: SignWord[] = $state([]);
  let progress: Progress = $state({});
  let note = $state("");
  let timer: ReturnType<typeof setTimeout>;

  // Searching by signing: the recording fills the same grid the field does. Nothing is spoken and
  // nothing is scored — this is a lookup, not an attempt. The camera is simply there the whole time
  // (user, 2026-09-24): now that the hands start a search by themselves there is no way to be in,
  // and so nothing to switch between — the learner types a word or signs one.
  const camera = useCamera();
  const lexicon = $derived(camera.lexicon);
  let searching = $state(false);

  $effect(() => {
    progress = load();
  });

  const clips = $derived(new Map(lexicon?.signs.map((each) => [each.sign, each.references]) ?? []));
  const label = (each: SignWord) => each.word ?? word(each.sign);

  function search(text: string) {
    clearTimeout(timer);
    query = text;
    timer = setTimeout(async () => {
      results = text.trim() ? await fetchSearch(text) : [];
    }, WAIT);
  }

  // The hands start and end the recording, as the voice does in Träna (user, 2026-09-24): raising
  // them to sign records, lowering them searches. Nothing is pressed here either, and a learner
  // reading their results never has a recording started under them, because the hands have to leave
  // the picture before the next one can begin.
  let eyes = $state(watching());

  $effect(() => {
    const tracker = camera.tracker;
    if (!tracker) return;
    const watch = setInterval(() => {
      if (searching) return; // the last sign is still being looked up; the next one waits for it
      eyes = see(eyes, hands(tracker.latest));
      if (eyes.phase === "signing" && !camera.recording) return void record();
      // Hands held up and never lowered would record until the page is left, so a recording is cut
      // at the length one sign is allowed to be, which is what it is going to be searched for.
      const tooLong = lexicon && Date.now() - began > lexicon.maxSeconds * 1000;
      if (eyes.phase === "done" || (camera.recording && tooLong)) (eyes = watching()), void finish();
    }, LOOK);
    return () => {
      clearInterval(watch);
      (eyes = watching()), (camera.recording = false);
    };
  });

  let began = 0; // when the running recording started, for the length one sign is allowed to be

  function record() {
    if (!camera.tracker) return;
    camera.tracker.startRecording();
    began = Date.now();
    (camera.recording = true), (note = "");
  }

  async function finish() {
    if (!camera.recording) return;
    camera.recording = false;
    const taken = await camera.tracker?.stopRecording();
    if (!taken || taken.frames.length < 2) return void (note = "Inspelningen är tom.");
    searching = true;
    try {
      const found = await searchBySign(taken.frames, hand() ?? "right", camera.video!, lexicon!.fps);
      (results = found.words), (note = found.note), (query = "");
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

  const picked = (each: SignWord) => !!progress[each.sign];
</script>

<!-- The picture first and the field under it (user, 2026-09-24): the two ways in read as one, and
     the field's placeholder is where both are said, so the page needs no line of instructions. -->
<section class="signing">
  <CameraView {camera} />
  <!-- the picture says whether it is recording, so the only line here is the wait after a sign -->
  {#if searching}<p class="dim">Söker ...</p>{/if}
</section>

<input
  type="search"
  placeholder="Teckna ordet, eller sök efter ord eller teman"
  value={query}
  oninput={(event) => search(event.currentTarget.value)}
/>

{#if note}
  <p class="dim note">{note}</p>
{/if}

{#if results.length}
  <p class="dim found">{results.length} tecken</p>

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
  /* the picture is the middle of the page, with the field under it */
  .signing {
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: 100%;
    max-width: 360px;
    margin: 0 auto 16px;
    text-align: center;
  }

  .note {
    margin-top: 16px;
  }

  .found {
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
