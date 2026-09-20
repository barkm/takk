<script lang="ts">
  import Recorder from "$lib/Recorder.svelte";
  import Verdict from "$lib/Verdict.svelte";
  import {
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
  import { chosenWords, load, loadChosen, record, save, saveChosen, session, type Progress } from "$lib/progress";
  import { page } from "$app/state";

  let lexicon = $state<Lexicon | null>(null);
  let packs: Pack[] = $state([]);
  let chosen: string[] = $state([]);
  let progress: Progress = $state({});
  let today: PackWord[] = $state([]);
  let at = $state(0);
  let attempt: Attempt | null = $state(null);
  let note = $state("");
  let correct = $state(0);

  $effect(() => {
    Promise.all([fetchLexicon(), fetchPacks()])
      .then(([loadedLexicon, loadedPacks]) => {
        (lexicon = loadedLexicon), (packs = loadedPacks);
        progress = load();
        chosen = loadChosen(packs);
        restart();
      })
      .catch(() => (note = "Servern svarar inte. Starta den med uv run takk."));
  });

  // `?days=1` picks the session as it will look a day from now, to try the spaced repetition without
  // waiting for it. Only what counts as due moves; an attempt is still recorded at the real time.
  const ahead = $derived(Number(new URLSearchParams(page.url.search).get("days")) || 0);

  // Today's signs are picked once, not derived: a scored attempt changes the progress they come from.
  function restart() {
    today = session(chosenWords(packs, chosen), progress, 5, Date.now() + ahead * 24 * 60 * 60 * 1000);
    (at = 0), (correct = 0), (attempt = null), (note = "");
  }

  function choose(name: string, on: boolean) {
    chosen = on ? [...chosen, name] : chosen.filter((other) => other !== name);
    saveChosen(chosen);
    restart();
  }

  const current = $derived(today[at]);
  const shown = $derived(current ? (current.word ?? word(current.sign)) : "");
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

{#snippet picker()}
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
            <span class="dim">{pack.words.length} tecken{pack.kind === "category" ? ", ämnesområde i lexikonet" : ""}</span>
          </label>
        </li>
      {/each}
    </ul>
  </details>
{/snippet}

{#if lexicon && current}
  <section class="card">
    <h1>Dagens pass</h1>
    <p class="dim">
      Tecken {at + 1} av {today.length}. Titta på klippet och teckna ordet.
      {#if ahead}Passet visas som det ser ut om {ahead} dagar.{/if}
    </p>
    <h2>{shown}</h2>
    <div class="videos">
      {#each references as clip (clip)}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video src={referenceUrl(clip)} autoplay loop muted playsinline controls></video>
      {/each}
    </div>
    {@render picker()}
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
      {today.length
        ? `Klart! ${correct} av ${today.length} tecken rätt.`
        : "Inget att öva just nu. Välj fler ord, eller kom tillbaka när dagens tecken ska repeteras."}
    </p>
    {@render picker()}
  </section>
{:else}
  <section class="card">
    <h1>Dagens pass</h1>
    <p class="dim">{note || "Laddar lexikonet …"}</p>
  </section>
{/if}

<style>
  ul {
    max-height: 40vh; /* the lexicon's categories are 58 of them */
    overflow-y: auto;
    margin: 8px 0;
    padding-left: 20px;
  }
</style>
