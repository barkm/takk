<script lang="ts">
  import Recorder from "$lib/Recorder.svelte";
  import Verdict from "$lib/Verdict.svelte";
  import {
    fetchForm,
    fetchLexicon,
    fetchPacks,
    referenceUrl,
    word,
    type Attempt,
    type Lexicon,
    type Pack,
    type PackWord,
    type Sign,
  } from "$lib/api";
  import { chosenWords, load, loadChosen, record, save, saveChosen, type Progress } from "$lib/progress";

  // New words (step 9 of ROADMAP-takk.md): the only way vocabulary grows. Every card here is a word
  // never practised, so it is always taught — its clip and the lexicon's description of the form are
  // on screen while it is signed. An accepted word goes into the first Leitner box and is repeated in
  // the story; a missed one is simply signed again, since there is nothing to test yet.
  const SIZE = 5; // new words in one session

  let lexicon = $state<Lexicon | null>(null);
  let packs: Pack[] = $state([]);
  let chosen: string[] = $state([]);
  let progress: Progress = $state({});
  let queue: PackWord[] = $state([]); // the words left, the current one first
  let started = $state(false);
  let taken = $state(0); // how many of this session's words are done
  let attempt: Attempt | null = $state(null);
  let note = $state("");
  let form = $state(""); // the lexicon's description of the current sign
  let recorder: ReturnType<typeof Recorder> | undefined = $state();

  $effect(() => {
    Promise.all([fetchLexicon(), fetchPacks()])
      .then(([loadedLexicon, loadedPacks]) => {
        (lexicon = loadedLexicon), (packs = loadedPacks);
        progress = load();
        chosen = loadChosen(packs);
      })
      .catch(() => (note = "Servern svarar inte. Starta den med uv run takk."));
  });

  /** The word on a card, which is not the name of the sign that scores it when several words share
   * one sign form: "blå" is scored by `sts:öga-02636`, the lowest entry of that form. */
  const label = (each: PackWord) => each.word ?? word(each.sign);

  const fresh = $derived(chosenWords(packs, chosen).filter((each) => !progress[each.sign]));

  function start() {
    queue = fresh.slice(0, SIZE);
    (taken = 0), (started = true), (attempt = null), (note = "");
  }

  function choose(name: string, on: boolean) {
    chosen = on ? [...chosen, name] : chosen.filter((other) => other !== name);
    saveChosen(chosen);
  }

  const current = $derived(queue[0]);
  const shown = $derived(current ? label(current) : "");
  const references = $derived(lexicon?.signs.find((each) => each.sign === current?.sign)?.references ?? []);
  const sentence: Sign[] = $derived(current ? [{ sign: current.sign, references, spoken: shown }] : []);
  const labels = $derived(current ? { [current.sign]: shown } : {});

  $effect(() => {
    const entryId = current?.id;
    form = "";
    if (entryId) fetchForm(entryId).then((described) => (form = described));
  });

  function scored(scoredAttempt: Attempt | null, told: string) {
    (attempt = scoredAttempt), (note = told);
    // Nothing is pressed between cards: a recording that could not be used, and a sign that was not
    // recognised, are both simply recorded again, with the clip still on screen to sign from.
    const judged = scoredAttempt?.signs ?? [];
    if (!judged.length || !judged[0].usable || !judged[0].correct) return void recorder?.arm(true);
    progress = record(progress, { ...current, word: shown }, true);
    save(progress);
    (queue = queue.slice(1)), (taken += 1), (attempt = null), (note = "");
  }
</script>

{#if !lexicon}
  <section class="card">
    <h1>Nya ord</h1>
    <p class="dim">{note || "Laddar lexikonet …"}</p>
  </section>
{:else if !started}
  <section class="card">
    <h1>Nya ord</h1>
    <p class="dim">
      {SIZE} nya tecken ur orden du valt. Du ser klippet och tecknar efter det. Ett tecken du får rätt
      hamnar i första lådan och kommer tillbaka i <a href="/saga">sagan</a>.
    </p>
    <details>
      <summary>Övar på: {chosen.join(", ") || "inget valt"}</summary>
      <ul>
        {#each packs as pack (pack.name)}
          <li>
            <label>
              <input
                type="checkbox"
                checked={chosen.includes(pack.name)}
                onchange={(event) => choose(pack.name, event.currentTarget.checked)}
              />
              {pack.name}
              <span class="dim">{pack.words.length} tecken</span>
            </label>
          </li>
        {/each}
      </ul>
    </details>
    <button onclick={start} disabled={!fresh.length}>Börja</button>
    {#if !fresh.length}
      <p class="dim">Inga nya ord kvar bland de valda orden. Välj fler.</p>
    {/if}
  </section>
{:else if current}
  <section class="card">
    <p class="dim">{taken} av {taken + queue.length} tecken klara. Säg ordet högt medan du tecknar det.</p>
    <h1>{shown}</h1>
    {#if references.length}
      <!-- svelte-ignore a11y_media_has_caption -->
      <video src={referenceUrl(references[0])} autoplay loop muted playsinline controls></video>
    {/if}
    {#if form}<p class="form">{form}</p>{/if}
  </section>
  <Recorder bind:this={recorder} {sentence} {lexicon} onattempt={scored} />
  <Verdict {attempt} {note} {labels} />
{:else}
  <section class="card">
    <h1>Klart!</h1>
    <p>{taken} nya tecken i första lådan. Repetera dem i <a href="/saga">sagan</a>.</p>
    <button onclick={start} disabled={!fresh.length}>Fler nya ord</button>
  </section>
{/if}

<style>
  video {
    display: block;
    width: 100%;
    max-width: 560px;
    border-radius: 8px;
  }

  .form {
    margin: 8px 0;
    font-style: italic;
  }

  ul {
    max-height: 40vh; /* the lexicon's categories are 58 of them */
    overflow-y: auto;
    margin: 8px 0;
    padding-left: 20px;
  }
</style>
