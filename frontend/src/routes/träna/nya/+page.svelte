<script lang="ts">
  import Recorder from "$lib/Recorder.svelte";
  import Verdict from "$lib/Verdict.svelte";
  import { fetchForm, referenceUrl, word, type Attempt, type SignWord, type Sign } from "$lib/api";
  import { useCamera } from "$lib/camera.svelte";
  import { fresh, load, record, save, type Progress } from "$lib/progress";

  // Nya ord (steps 9 and 12 of ROADMAP-takk.md): where a sign the learner picked in Tecken is taught
  // for the first time. Every card is a word never practised, so it is always taught — its clip and
  // the lexicon's description of the form are on screen while it is signed. An accepted word goes
  // into the first Leitner box and is repeated in the story; a missed one is simply signed again,
  // since there is nothing to test yet.
  const SIZE = 5; // new words in one session

  const camera = useCamera(); // the camera of the Träna layout, which both modes share
  const lexicon = $derived(camera.lexicon);
  let progress: Progress = $state({});
  let queue: SignWord[] = $state([]); // the words left, the current one first
  let started = $state(false);
  let taken = $state(0); // how many of this session's words are done
  let attempt: Attempt | null = $state(null);
  let note = $state("");
  let form = $state(""); // the lexicon's description of the current sign
  let recorder: ReturnType<typeof Recorder> | undefined = $state();

  $effect(() => {
    progress = load();
  });

  /** The word on a card, which is not the name of the sign that scores it when several words share
   * one sign form: "blå" is scored by `sts:öga-02636`, the lowest entry of that form. */
  const label = (each: SignWord) => each.word ?? word(each.sign);

  const waiting = $derived(fresh(progress)); // picked in Tecken and not practised yet, oldest first

  function start() {
    queue = waiting.slice(0, SIZE);
    (taken = 0), (started = true), (attempt = null), (note = "");
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
    <p class="dim">{note || "Laddar lexikonet ..."}</p>
  </section>
{:else if !started}
  <section class="card">
    <h1>Nya ord</h1>
    <p class="dim">
      {SIZE} nya tecken av dem du valt. Du ser klippet och tecknar efter det.
    </p>
    <button onclick={start} disabled={!waiting.length}>Börja</button>
    {#if !waiting.length}
      <p class="dim">Inga nya tecken valda. Välj några under <a href="/sök">Sök</a>.</p>
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
  <Recorder bind:this={recorder} {sentence} onattempt={scored} />
  <Verdict {attempt} {note} {labels} />
{:else}
  <section class="card">
    <h1>Klart!</h1>
    <p>{taken} nya tecken klara. <a href="/träna/repetera">Repetera</a> dem när du vill.</p>
    <button onclick={start} disabled={!waiting.length}>Fler nya ord</button>
  </section>
{/if}

<style>
  /* the lexicon's own shape, held before the clip loads so the card does not jump when it arrives */
  video {
    display: block;
    width: 100%;
    max-width: 560px;
    aspect-ratio: 4 / 3;
    object-fit: cover;
    background: #000;
    border-radius: 12px;
  }

  .form {
    margin: 8px 0;
    font-style: italic;
  }
</style>
