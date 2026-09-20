<script lang="ts">
  import Recorder from "$lib/Recorder.svelte";
  import Verdict from "$lib/Verdict.svelte";
  import {
    fetchLexicon,
    fetchForm,
    fetchPacks,
    fetchSentence,
    referenceUrl,
    word,
    type Attempt,
    type Lexicon,
    type Pack,
    type PackWord,
    type Sign,
  } from "$lib/api";
  import {
    advance,
    chosenWords,
    DEFAULTS,
    load,
    loadChosen,
    loadSetting,
    practisedToday,
    record,
    save,
    saveChosen,
    saveSetting,
    session,
    turn,
    type Progress,
  } from "$lib/progress";
  import { page } from "$app/state";

  let lexicon = $state<Lexicon | null>(null);
  let packs: Pack[] = $state([]);
  let chosen: string[] = $state([]);
  let size = $state(DEFAULTS.size); // words in a pass, the learner's own number
  let needed = $state(DEFAULTS.accepts); // accepted attempts before a word is finished
  let progress: Progress = $state({});
  let queue: PackWord[] = $state([]); // the turns left in the pass, the current one first
  let accepts: Record<string, number> = $state({}); // accepted attempts per sign, in this pass alone
  let taught: string[] = $state([]); // the signs whose clip has already been shown in this pass
  let missed: string[] = $state([]); // the signs missed in this pass, which do not move up when finished
  let answered = $state(false); // whether this turn's answer is in, so a second recording is a retry
  let taken: PackWord[] = $state([]); // the words of the turn on screen, in the order they are signed
  let said = $state(""); // the Swedish sentence they are signed in, empty when the turn is one word
  let writing = $state(false); // whether the sentence is still being written
  let total = $state(0);
  let attempt: Attempt | null = $state(null);
  let note = $state("");
  let peeked = $state(false);
  let form = $state(""); // the lexicon's description of the current sign, fetched per card

  $effect(() => {
    Promise.all([fetchLexicon(), fetchPacks()])
      .then(([loadedLexicon, loadedPacks]) => {
        (lexicon = loadedLexicon), (packs = loadedPacks);
        progress = load();
        chosen = loadChosen(packs);
        (size = loadSetting("size")), (needed = loadSetting("accepts"));
        restart();
      })
      .catch(() => (note = "Servern svarar inte. Starta den med uv run takk."));
  });

  // `?days=1` picks the session as it will look a day from now, to try the spaced repetition without
  // waiting for it. Only what counts as due moves; an attempt is still recorded at the real time.
  const ahead = $derived(Number(new URLSearchParams(page.url.search).get("days")) || 0);

  // The pass's signs are picked once, not derived: a scored attempt changes the progress they come from.
  function restart() {
    queue = session(chosenWords(packs, chosen), progress, size, Date.now() + ahead * 24 * 60 * 60 * 1000);
    (total = queue.length), (accepts = {}), (taught = []), (missed = []);
    begin();
  }

  /** Whether a word is being met for the first time, which is the one turn that is not spoken. */
  const unmet = (each: PackWord) => !progress[each.sign] && !taught.includes(each.sign);

  // Start a turn on the front of the queue. A word met for the first time is taught by its clip and
  // signed on its own; every other turn is practised the way TAKK is used, inside a sentence said
  // aloud — of several words when the queue offers them, of one when it does not. A sentence the
  // server could not write leaves the words to be signed alone and unspoken, which is the fallback
  // that needs no model at all.
  async function begin() {
    (attempt = null), (note = ""), (peeked = false), (answered = false), (said = "");
    const words = turn(queue, progress);
    taken = words.slice(0, 1);
    if (!words.length || unmet(words[0])) return;
    writing = true;
    const written = await fetchSentence(words.map((each) => each.sign)).finally(() => (writing = false));
    if (!written.sentence) return;
    const order = new Map(words.map((each) => [each.sign, each]));
    (said = written.sentence), (taken = written.signs.map((sign) => order.get(sign)!));
  }

  function setSetting(name: "size" | "accepts", value: number) {
    saveSetting(name, value);
    (size = loadSetting("size")), (needed = loadSetting("accepts"));
    restart();
  }

  function choose(name: string, on: boolean) {
    chosen = on ? [...chosen, name] : chosen.filter((other) => other !== name);
    saveChosen(chosen);
    restart();
  }

  const current = $derived(taken[0]);
  const alone = $derived(taken.length < 2); // one word, so the whole recording is the attempt
  // What the lexicon says the hands do, which is the only teaching text besides the clip. It belongs
  // to a word being taught, so a sentence of words already known needs none.
  $effect(() => {
    const entryId = alone ? current?.id : undefined;
    form = "";
    if (entryId) fetchForm(entryId).then((described) => (form = described));
  });

  // Whether another pass would have anything in it, which is what makes the summary offer one.
  const waiting = $derived(session(chosenWords(packs, chosen), progress, 1, Date.now() + ahead * 24 * 60 * 60 * 1000).length > 0);  // prettier-ignore
  const label = (each: PackWord) => each.word ?? word(each.sign);
  const shown = $derived(current ? label(current) : "");
  const labels = $derived(Object.fromEntries(taken.map((each) => [each.sign, label(each)])));
  const finished = $derived(Object.values(accepts).filter((count) => count >= needed).length);
  // The tutorial belongs to the very first time a word is met: a word never practised, on its first
  // turn of this pass. Every later turn is a test, so the clip only follows the verdict. Looking it up
  // first is allowed but does not count as recalled. A sentence is only ever made of words already
  // practised on their own, so it is never a tutorial.
  const teaching = $derived(!said && !!current && unmet(current));
  const showClip = $derived(alone && (teaching || peeked || !!attempt));
  const clipsOf = (sign: string) => lexicon?.signs.find((each) => each.sign === sign)?.references ?? [];
  const references = $derived(current ? clipsOf(current.sign) : []);
  const sentence: Sign[] = $derived(taken.map((each) => ({ sign: each.sign, references: clipsOf(each.sign) })));

  function scored(scoredAttempt: Attempt | null, told: string) {
    (attempt = scoredAttempt), (note = told);
    // A sentence the recording could not be split into its signs is no answer for any of its words,
    // as an unusable recording is none for one: the boxes are not moved by a bad split.
    if (!scoredAttempt?.signs.length || scoredAttempt.signs.some((judged) => !judged.usable)) return;
    if (answered) return; // a second recording of the same card is a retry: a verdict, but not an answer
    answered = true;
    for (const judged of scoredAttempt.signs) settle(judged.sign, !!judged.correct && !peeked);
    save(progress);
  }

  /** Record one sign's verdict. The box only moves when the word's fate is settled: it is finished,
   * or it was just missed and drops back to the first box. An accept that leaves the word short of
   * `needed` keeps it in the pass and changes nothing. A word missed anywhere in the pass stays in
   * the first box when it finishes, rather than climbing out of the box the miss put it in. */
  function settle(sign: string, ok: boolean) {
    taught = [...taught, sign];
    const count = (accepts[sign] ?? 0) + (ok ? 1 : 0);
    accepts = { ...accepts, [sign]: count };
    if (!ok && !missed.includes(sign)) missed = [...missed, sign];
    if (!ok) progress = record(progress, sign, false, labels[sign]);
    else if (count >= needed) progress = record(progress, sign, !missed.includes(sign), labels[sign]);
  }

  function next() {
    queue = advance(queue, taken, accepts, needed);
    begin();
  }
</script>

{#snippet picker()}
  <details>
    <summary>Övar på: {chosen.join(", ") || "inget valt"}</summary>
    <label class="number">
      Tecken per pass:
      <input type="number" min="1" max="50" value={size} onchange={(e) => setSetting("size", Number(e.currentTarget.value))} />
    </label>
    <label class="number">
      Rätt per tecken:
      <input type="number" min="1" max="10" value={needed} onchange={(e) => setSetting("accepts", Number(e.currentTarget.value))} />
    </label>
    <p class="dim">
      Tecken som ska repeteras kommer först, och nya tecken fyller på upp till {size}. Ett tecken är klart
      när det har godkänts {needed} gånger, och flyttas då upp en låda. {practisedToday(progress)} tecken
      övade idag.
    </p>
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
      {finished} av {total} tecken klara.
      {#if writing}Skriver en mening ...{:else if said}Säg meningen högt medan du tecknar {taken.length > 1 ? 'orden' : 'ordet'} i fetstil.{:else if teaching}Nytt tecken: titta på klippet och teckna det.{:else}Repetition.{/if}
      {#if needed > 1 && alone}Godkänt {accepts[current.sign] ?? 0} av {needed} gånger.{/if}
      {#if peeked && !attempt}Du tog fram tecknet, så det räknas inte som godkänt.{/if}
      {#if ahead}Passet visas som det ser ut om {ahead} dagar.{/if}
    </p>
    {#if said}
      <h2 class="sentence">
        {#each said.split(new RegExp(`\\b(${taken.map((each) => labels[each.sign]).join("|")})\\b`, "i")) as part, at (at)}
          {#if at % 2}<strong>{part}</strong>{:else}{part}{/if}
        {/each}
      </h2>
    {:else}
      <h2>{shown}</h2>
    {/if}
    {#if showClip}
      <div class="videos">
        {#each references as clip (clip)}
          <!-- svelte-ignore a11y_media_has_caption -->
          <video src={referenceUrl(clip)} autoplay loop muted playsinline controls></video>
        {/each}
      </div>
      {#if form}<p class="form">{form}</p>{/if}
    {:else if alone}
      <p class="dim">Teckna ordet ur minnet. Klippet visas när du har spelat in.</p>
      <button class="secondary" onclick={() => (peeked = true)}>Jag kommer inte ihåg — visa tecknet</button>
    {/if}
    {@render picker()}
  </section>
  <Recorder {sentence} {lexicon} spoken={!!said} onattempt={scored} />
  <Verdict {attempt} {note} {labels} />
  {#if attempt?.signs.length && alone && word(current.sign) !== shown}
    <section class="card">
      <p class="dim">
        Tecknet för {shown} har samma form som {word(current.sign)} i lexikonet, så det är det namnet
        modellen räknar med.
      </p>
    </section>
  {/if}
  {#if attempt?.signs.length}
    <section class="card">
      <button onclick={next}>{advance(queue, taken, accepts, needed).length ? "Nästa" : "Avsluta passet"}</button>
    </section>
  {/if}
{:else if lexicon}
  <section class="card">
    <h1>Dagens pass</h1>
    <p>
      {total
        ? `Klart! ${total} tecken igenom, ${practisedToday(progress)} tecken övade idag.`
        : "Inget att öva just nu. Välj fler ord, eller kom tillbaka när dagens tecken ska repeteras."}
    </p>
    {#if waiting}
      <button onclick={restart}>Ett pass till</button>
    {:else if total}
      <p class="dim">Inget mer att öva idag — kom tillbaka i morgon.</p>
    {/if}
    {@render picker()}
  </section>
{:else}
  <section class="card">
    <h1>Dagens pass</h1>
    <p class="dim">{note || "Laddar lexikonet …"}</p>
  </section>
{/if}

<style>
  .number input {
    width: 4em;
  }

  .form {
    margin: 8px 0;
    font-style: italic;
  }

  .sentence {
    font-weight: 400; /* the key words are the bold ones, so the sentence around them is not */
  }

  ul {
    max-height: 40vh; /* the lexicon's categories are 58 of them */
    overflow-y: auto;
    margin: 8px 0;
    padding-left: 20px;
  }
</style>
