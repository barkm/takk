<script lang="ts">
  import { about } from "$lib/about";
  import {
    fetchChips,
    fetchSearch,
    lexiconUrl,
    searchBySign,
    word,
    type Chip,
    type SignWord,
  } from "$lib/api";
  import CameraView from "$lib/Camera.svelte";
  import Loading from "$lib/Loading.svelte";
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
  let searching = $state(false); // a sign is being looked up
  let finding = $state(false); // a word is being looked up

  $effect(() => {
    progress = load();
  });

  // The chips under the field (step 23 of ROADMAP-takk.md): a few sets of words the model fitted to
  // what the learner said on Om dig, each pressed to fill the grid as a search does. They are kept
  // with what they were made for and asked for again only when that changes, since every ask is a
  // call to the model.
  // ponytail: not asked for again as the vocabulary grows; add a refresh when the chips run dry.
  const CHIPS = "takk.chips";
  let chips: Chip[] = $state([]);
  let chip = $state(""); // the label of the chip whose words fill the grid
  let suggesting = $state(false);

  $effect(() => {
    const said = about();
    if (!said) return;
    const made = JSON.stringify(said);
    try {
      const kept = JSON.parse(localStorage.getItem(CHIPS) ?? "null");
      if (kept?.made === made) return void (chips = kept.chips);
    } catch {
      // an unreadable store asks the model again
    }
    suggesting = true;
    fetchChips(said.level, said.context, Object.values(load()).map((each) => each.word))
      .then((found) => {
        chips = found;
        if (found.length) localStorage.setItem(CHIPS, JSON.stringify({ made, chips: found }));
      })
      .catch(() => {})
      .finally(() => (suggesting = false));
  });

  function press(pressed: Chip) {
    clearTimeout(timer);
    (chip = pressed.label), (results = pressed.words), (query = ""), (note = "");
  }

  const clips = $derived(new Map(lexicon?.signs.map((each) => [each.sign, each.references]) ?? []));
  const label = (each: SignWord) => each.word ?? word(each.sign);

  function search(text: string) {
    clearTimeout(timer);
    (query = text), (chip = "");
    timer = setTimeout(async () => {
      finding = true;
      results = text.trim() ? await fetchSearch(text).finally(() => (finding = false)) : [];
      finding = false;
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
      (results = found.words), (note = found.note), (query = ""), (chip = "");
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
<!-- nothing is written about the picture: not what it is doing, and not that a sign is being looked
     up — the loader under the field runs while it is (user, 2026-09-26), and the results arriving
     say it is done -->
<section class="signing">
  <CameraView {camera} />
</section>

<input
  type="search"
  placeholder="Teckna ordet, eller sök efter ord eller teman"
  value={query}
  oninput={(event) => search(event.currentTarget.value)}
/>
<!-- its space is held whether or not anything is being looked up, so nothing moves when it runs -->
<div class="wait">{#if searching || finding || suggesting}<Loading label="Söker" />{/if}</div>

{#if chips.length}
  <div class="chips">
    {#each chips as each (each.label)}
      <button class:secondary={chip !== each.label} aria-pressed={chip === each.label} onclick={() => press(each)}>
        {each.label}
      </button>
    {/each}
  </div>
{/if}

{#if note}
  <p class="dim note">{note}</p>
{/if}

{#if results.length}
  <p class="dim found">{results.length} tecken</p>

  <ul class="grid">
    {#each results as each (each.sign)}
      {@const clip = clips.get(each.sign)?.[0]}
      <li class="tile" class:on={picked(each)}>
        {#if clip}
          <!-- svelte-ignore a11y_media_has_caption -->
          <!-- Still until the tile is pointed at, with the fragment seeking 0.6 s in so the tile
               shows the sign being made rather than the signer still at rest. Thirty clips playing
               at once, beside the camera and the landmarker, is more video than a browser will
               decode: the ones it gives up on stay black, and the camera can be the one it gives
               up on. -->
          <video src="{clip}#t=0.6" muted loop playsinline preload="metadata"></video>
        {:else}
          <span class="noclip dim">inget klipp</span>
        {/if}
        <span class="name">
          <a class="word" href={lexiconUrl(each.id)} target="_blank" rel="noopener">{label(each)}</a>
          <span class="mark" aria-hidden="true">{picked(each) ? "Vald" : "Lägg till"}</span>
        </span>
        <!-- The whole tile picks the word, except the word itself, which links to the lexicon as on
             Tecken. A link cannot sit inside a button, so the button lies over the tile instead of
             holding it, and it is the button the pointer is over that plays the clip. -->
        <button
          class="pick"
          aria-label={label(each)}
          aria-pressed={picked(each)}
          onclick={() => pick(each)}
          onpointerenter={(event) => event.currentTarget.parentElement?.querySelector("video")?.play()}
          onpointerleave={(event) => event.currentTarget.parentElement?.querySelector("video")?.pause()}
        ></button>
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
    max-width: 440px;
    margin: 0 auto 16px;
    text-align: center;
  }

  /* as wide as the picture above it, so the two ways in read as one */
  input[type="search"] {
    display: block;
    max-width: 440px;
    margin: 0 auto;
  }

  .wait {
    height: 2px;
    margin-top: 6px;
  }

  .note {
    margin-top: 16px;
  }

  .chips {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    max-width: 440px;
    margin: 14px auto 0;
  }

  .chips button {
    padding: 8px 14px;
    font-size: 14px;
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
    position: relative;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid transparent; /* grey like every tile; only a picked one is outlined */
    border-radius: var(--radius);
    font-weight: 500;
  }

  .tile:hover {
    background: var(--raised);
  }

  .pick {
    position: absolute;
    inset: 0;
    padding: 0;
    border: 0;
    border-radius: inherit;
    background: transparent;
  }

  .pick:hover:not(:disabled) {
    filter: none;
  }

  /* above the button lying over the tile, so it is the one word that is not a pick */
  .word {
    position: relative;
    z-index: 1;
    color: inherit;
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
