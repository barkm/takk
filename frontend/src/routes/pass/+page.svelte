<script lang="ts">
  import Recorder from "$lib/Recorder.svelte";
  import Verdict from "$lib/Verdict.svelte";
  import { fetchLexicon, fetchPacks, referenceUrl, type Attempt, type Lexicon, type PackWord, type Sign } from "$lib/api";
  import { load, record, save, session, type Progress } from "$lib/progress";

  let lexicon = $state<Lexicon | null>(null);
  let progress: Progress = $state({});
  let today: PackWord[] = $state([]);
  let at = $state(0);
  let attempt: Attempt | null = $state(null);
  let note = $state("");
  let correct = $state(0);

  $effect(() => {
    Promise.all([fetchLexicon(), fetchPacks()])
      .then(([loaded, packs]) => {
        lexicon = loaded;
        progress = load();
        today = session(packs.flatMap((pack) => pack.words), progress);
      })
      .catch(() => (note = "Servern svarar inte. Starta den med uv run takk."));
  });

  const current = $derived(today[at]);
  // The Recorder scores a sentence, so one word is a sentence of one sign, with the sign's lexicon clips.
  const references = $derived(lexicon?.signs.find((sign) => sign.sign === current?.sign)?.references ?? []);
  const sentence: Sign[] = $derived(current ? [{ sign: current.sign, references }] : []);

  function scored(scoredAttempt: Attempt | null, said: string) {
    (attempt = scoredAttempt), (note = said);
    const judged = scoredAttempt?.signs[0];
    if (!judged?.usable) return; // a recording that could not be used is not an answer either way
    if (judged.correct) correct += 1;
    progress = record(progress, judged.sign, !!judged.correct);
    save(progress);
  }

  function next() {
    (at += 1), (attempt = null), (note = "");
  }
</script>

{#if lexicon && current}
  <section class="card">
    <h1>Dagens pass</h1>
    <p class="dim">Tecken {at + 1} av {today.length}. Titta på klippet och teckna ordet.</p>
    <h2>{current.word}</h2>
    <div class="videos">
      {#each references as clip (clip)}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video src={referenceUrl(clip)} autoplay loop muted playsinline controls></video>
      {/each}
    </div>
  </section>
  <Recorder {sentence} {lexicon} onattempt={scored} />
  <Verdict {attempt} {note} />
  {#if attempt?.signs.length}
    <section class="card">
      <button onclick={next}>{at + 1 < today.length ? "Nästa tecken" : "Avsluta passet"}</button>
    </section>
  {/if}
{:else if lexicon}
  <section class="card">
    <h1>Dagens pass</h1>
    <p>
      {today.length ? `Klart! ${correct} av ${today.length} tecken rätt.` : "Inget att öva just nu — kom tillbaka senare."}
    </p>
    <a href="/">Till fri övning</a>
  </section>
{:else}
  <section class="card">
    <h1>Dagens pass</h1>
    <p class="dim">{note || "Laddar lexikonet …"}</p>
  </section>
{/if}
