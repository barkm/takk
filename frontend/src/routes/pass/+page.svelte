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
    DEFAULTS,
    known,
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
  let started = $state(false); // whether a pass is running: the page configures one first, then starts it
  let mode = $state<"dagens" | "alla">("dagens"); // the kind of pass, chosen before it starts
  let form = $state(""); // the lexicon's description of the current sign, fetched per card
  let recorder: ReturnType<typeof Recorder> | undefined = $state(); // to arm the same card again

  const PAUSE = 2000; // ms to read the verdict before an accepted card gives way to the next one
  let ticket = 0; // bumped by every card and every verdict, so a pending advance can tell it is stale

  $effect(() => {
    Promise.all([fetchLexicon(), fetchPacks()])
      .then(([loadedLexicon, loadedPacks]) => {
        (lexicon = loadedLexicon), (packs = loadedPacks);
        progress = load();
        chosen = loadChosen(packs);
        (size = loadSetting("size")), (needed = loadSetting("accepts"));
      })
      .catch(() => (note = "Servern svarar inte. Starta den med uv run takk."));
  });

  // `?days=1` picks the session as it will look a day from now, to try the spaced repetition without
  // waiting for it. Only what counts as due moves; an attempt is still recorded at the real time.
  const ahead = $derived(Number(new URLSearchParams(page.url.search).get("days")) || 0);

  // "alla" is the review pass: `size` words drawn from everything practised, weighted towards the low
  // boxes, instead of the due words and the new ones. It teaches nothing new, and `record` leaves a
  // word that was not due where it is, so it can only repair the boxes.
  const review = $derived(mode === "alla");
  const title = $derived(review ? "Repetera allt du kan" : "Dagens pass");

  // The pass's signs are picked once, not derived: a scored attempt changes the progress they come from.
  function restart() {
    queue = review
      ? known(packs, progress, size)
      : session(packs, chosen, progress, size, Date.now() + ahead * 24 * 60 * 60 * 1000);
    (total = queue.length), (accepts = {}), (taught = []), (missed = []), (started = true);
    begin();
  }

  /** The word on a card, which is not the name of the sign that scores it when several words share
   * one sign form: "blå" is scored by `sts:öga-02636`, the lowest entry of that form. */
  const label = (each: PackWord) => each.word ?? word(each.sign);

  /** Whether a word is being met for the first time, which is when its clip is shown. */
  const unmet = (each: PackWord) => !progress[each.sign] && !taught.includes(each.sign);

  // Start a turn on the front of the queue. A word met for the first time is taught by its clip and
  // signed on its own; every other turn is practised the way TAKK is used, inside a sentence said
  // aloud — of several words when the queue offers them, of one when it does not. A sentence the
  // server could not write leaves the words to be signed alone and unspoken, which is the fallback
  // that needs no model at all.
  async function begin() {
    ticket++; // a card of its own: an advance still pending from the last one is stale
    (attempt = null), (note = ""), (peeked = false), (answered = false), (said = "");
    const words = turn(queue, progress);
    taken = words.slice(0, 1);
    if (!words.length || unmet(words[0])) return;
    writing = true;
    // The words are the ones on the cards, not the names of the signs that score them: "blå" is
    // scored by sts:öga-02636, and a sentence about an eye is neither what is being practised nor
    // what the learner would say. They are offered rather than required — the head apart — so what
    // comes back is the ones the sentence uses, in the order they occur in it.
    const written = await fetchSentence(words.map(label)).finally(() => (writing = false));
    if (!written.sentence) return;
    const byWord = new Map(words.map((each) => [label(each), each]));
    (said = written.sentence), (taken = written.words.map((each) => byWord.get(each)!));
  }

  function setSetting(name: "size" | "accepts", value: number) {
    saveSetting(name, value);
    (size = loadSetting("size")), (needed = loadSetting("accepts"));
  }

  function choose(name: string, on: boolean) {
    chosen = on ? [...chosen, name] : chosen.filter((other) => other !== name);
    saveChosen(chosen);
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
  const waiting = $derived(review ? known(packs, progress, 1).length > 0 : session(packs, chosen, progress, 1, Date.now() + ahead * 24 * 60 * 60 * 1000).length > 0);  // prettier-ignore
  const shown = $derived(current ? label(current) : "");
  const labels = $derived(Object.fromEntries(taken.map((each) => [each.sign, label(each)])));
  // The cards of this turn by sign, each under the word it is shown as, which is what the boxes keep.
  const cards = $derived(Object.fromEntries(taken.map((each) => [each.sign, { ...each, word: label(each) }])));
  const finished = $derived(Object.values(accepts).filter((count) => count >= needed).length);
  const practised = $derived(Object.keys(progress).length); // what a review pass has to draw from
  // The tutorial belongs to the very first time a word is met: a word never practised, on its first
  // turn of this pass. Every later turn is a test, so the clip only follows the verdict. Looking it up
  // first is allowed but does not count as recalled. A sentence is only ever made of words already
  // practised on their own, so it is never a tutorial.
  const teaching = $derived(!said && !!current && unmet(current));
  const showClip = $derived(alone && (teaching || peeked || !!attempt));
  const clipsOf = (sign: string) => lexicon?.signs.find((each) => each.sign === sign)?.references ?? [];
  const references = $derived(current ? clipsOf(current.sign) : []);
  const sentence: Sign[] = $derived(taken.map((each) => ({ sign: each.sign, references: clipsOf(each.sign), spoken: label(each) })));  // prettier-ignore

  function scored(scoredAttempt: Attempt | null, told: string) {
    (attempt = scoredAttempt), (note = told);
    // A sentence the recording could not be split into its signs is no answer for any of its words,
    // as an unusable recording is none for one: the boxes are not moved by a bad split, and the card
    // is simply recorded again.
    const judged = scoredAttempt?.signs ?? [];
    if (!judged.length || judged.some((each) => !each.usable)) return void recorder?.arm(true);
    if (!answered) {
      // a second recording of the same card is a retry: a verdict, but not an answer
      answered = true;
      for (const each of judged) settle(each.sign, !!each.correct && !peeked);
      save(progress);
    }
    // Nothing is pressed between cards (user, 2026-09-21): an accepted card gives way to the next one
    // after a pause long enough to read the verdict, and a missed one is armed again at once, with its
    // verdict and its clip still on screen to sign from. The ticket says the learner has neither
    // recorded again nor moved on while the pause ran; comparing the attempts themselves would not,
    // since `$state` hands out a proxy of the one that was stored, never the object that came in.
    const mine = ++ticket;
    if (judged.every((each) => each.correct)) setTimeout(() => mine === ticket && next(), PAUSE);
    else recorder?.arm(true);
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
    if (!ok) progress = record(progress, cards[sign], false);
    else if (count >= needed) progress = record(progress, cards[sign], !missed.includes(sign));
  }

  function next() {
    queue = advance(queue, taken, accepts, needed);
    begin();
  }
</script>

{#snippet picker()}
  <fieldset>
    <legend>Sorts pass</legend>
    <label>
      <input type="radio" value="dagens" bind:group={mode} />
      Dagens pass <span class="dim">tecken som ska repeteras, och nya ur de valda orden</span>
    </label>
    <label>
      <input type="radio" value="alla" bind:group={mode} />
      Repetera allt du kan <span class="dim">bland tecken du redan övat, inga nya</span>
    </label>
  </fieldset>
  <label class="number">
    Tecken per pass:
    <input type="number" min="1" max="50" value={size} onchange={(e) => setSetting("size", Number(e.currentTarget.value))} />
  </label>
  <label class="number">
    Rätt per tecken:
    <input type="number" min="1" max="10" value={needed} onchange={(e) => setSetting("accepts", Number(e.currentTarget.value))} />
  </label>
  <p class="dim">
    {#if review}
      Passet tar {size} tecken bland de {practised} du har övat, oftast ur de låga lådorna. Ett tecken som
      ännu inte skulle repeteras flyttas inte upp, men faller tillbaka till första lådan om du missar det.
    {:else}
      Tecken som ska repeteras kommer först, oavsett vilka ord de kommer ur, och nya tecken ur de valda
      orden fyller på upp till {size}. Ett tecken är klart
      när det har godkänts {needed} gånger, och flyttas då upp en låda.
    {/if}
    {practisedToday(progress)} tecken övade idag.
  </p>
  {#if !review}
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
  {/if}
{/snippet}

{#if !lexicon}
  <section class="card">
    <h1>Dagens pass</h1>
    <p class="dim">{note || "Laddar lexikonet …"}</p>
  </section>
{:else if !started}
  <!-- A pass is configured first and started on purpose, rather than beginning the moment the page
       loads: the kind of pass, its length and what it draws from are all chosen here (user, 2026-09-21). -->
  <section class="card">
    <h1>{title}</h1>
    {@render picker()}
    <button onclick={restart} disabled={review && !practised}>Starta passet</button>
    {#if review && !practised}
      <p class="dim">Inga övade tecken än — börja med dagens pass.</p>
    {/if}
  </section>
{:else if current}
  <section class="card">
    <h1>{title}</h1>
    <p class="dim">
      {finished} av {total} tecken klara.
      {#if writing}Skriver en mening ...{:else if said}Säg meningen högt medan du tecknar {taken.length > 1 ? "orden" : "ordet"} i fetstil.{:else if teaching}Nytt tecken: titta på klippet, säg ordet högt och teckna det.{:else}Repetition. Säg ordet högt medan du tecknar det.{/if}
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
    {:else if !writing}
      <h2>{shown}</h2>
    {/if}
    {#if showClip}
      {#if references.length}
        <div class="videos">
          <!-- svelte-ignore a11y_media_has_caption -->
          <video src={referenceUrl(references[0])} autoplay loop muted playsinline controls></video>
        </div>
      {/if}
      {#if form}<p class="form">{form}</p>{/if}
    {:else if alone}
      <p class="dim">Teckna ordet ur minnet. Klippet visas när du har spelat in.</p>
      <button class="secondary" onclick={() => (peeked = true)}>Jag kommer inte ihåg — visa tecknet</button>
    {/if}
  </section>
  <Recorder bind:this={recorder} {sentence} {lexicon} onattempt={scored} />
  <Verdict {attempt} {note} {labels} brief />
  {#if attempt?.signs.length}
    <section class="card">
      <button onclick={next}>{advance(queue, taken, accepts, needed).length ? "Nästa" : "Avsluta passet"}</button>
    </section>
  {/if}
{:else}
  <section class="card">
    <h1>{title}</h1>
    <p>
      {#if total}
        Klart! {total} tecken igenom, {practisedToday(progress)} tecken övade idag.
      {:else}
        Inget att öva just nu. Välj fler ord, eller kom tillbaka när dagens tecken ska repeteras.
      {/if}
    </p>
    {#if waiting}
      <button onclick={restart}>Ett pass till</button>
    {:else if total}
      <p class="dim">Inget mer att öva idag — kom tillbaka i morgon, eller repetera det du redan kan.</p>
    {/if}
    <button class="secondary" onclick={() => (started = false)}>Ändra inställningar</button>
  </section>
{/if}

<style>
  fieldset {
    margin: 8px 0;
    padding: 8px 12px;
    border: 1px solid var(--line);
    border-radius: 8px;
  }

  fieldset label {
    display: block;
    padding: 2px 0;
  }

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
