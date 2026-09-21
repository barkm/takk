<script lang="ts">
  import Recorder from "$lib/Recorder.svelte";
  import {
    fetchLexicon,
    fetchStory,
    word,
    type Attempt,
    type Lexicon,
    type SignWord,
    type Sign,
    type StoryPart,
  } from "$lib/api";
  import { known, load, record, save, type Progress } from "$lib/progress";
  import { pieces as cut } from "$lib/text";

  // A story is told in parts, one recording each (step 12 of ROADMAP-takk.md): the learner reads a
  // part aloud and signs the words marked in it, the words turn green or red, and the story goes on
  // whatever the verdict was. Only words already practised are used — new ones belong to the daily
  // pass — and they are drawn towards the low boxes, so a story leans on the weak words.
  const PARTS = 6; // parts in a story, one recording each
  // How long a part may be recorded for: it is read aloud, so its length is its text and not its
  // signs — a wordy part with one sign would be cut off by the one sign's worth of seconds a card
  // gets. Swedish read aloud runs at about two words a second; the slack is for reading it at all.
  const PER_WORD = 0.7;
  const SLACK = 4;
  const SHOWN = 1800; // ms the coloured words stay up before the next part

  let lexicon = $state<Lexicon | null>(null);
  let progress: Progress = $state({});
  let story: StoryPart[] = $state([]);
  let cards: Record<string, SignWord> = $state({}); // the card behind each word of the story, by word
  let at = $state(0); // the part being signed
  let writing = $state(false);
  let attempt: Attempt | null = $state(null);
  let note = $state("");
  let verdicts: Record<string, boolean> = $state({}); // this part's words, lowercased, right or wrong
  let waiting: ReturnType<typeof setTimeout> | null = null; // the pause the coloured words are read in
  let told: { word: string; correct: boolean }[] = $state([]); // every word of the story, in order
  let recorder: ReturnType<typeof Recorder> | undefined = $state();

  $effect(() => {
    fetchLexicon()
      .then((loaded) => {
        lexicon = loaded;
        progress = load();
      })
      .catch(() => (note = "Servern svarar inte. Starta den med uv run takk."));
  });

  const label = (each: SignWord) => each.word ?? word(each.sign);
  // A word is written as the story's sentence has it, so "Mamma" opening a part is the same word as
  // "mamma" in the list: verdicts are kept under the lowercase word.
  const same = (text: string) => text.toLowerCase();
  const pool = $derived(known(progress, Infinity, () => 0)); // everything practised, for the count

  /** Write a story over the words this learner knows, weakest first. Twice as many words are offered
   * as there are parts, so the model has something to choose from in every part. */
  async function begin() {
    (story = []), (told = []), (at = 0), (verdicts = {}), (attempt = null), (note = "");
    const words = known(progress, PARTS * 2);
    cards = Object.fromEntries(words.map((each) => [label(each), each]));
    writing = true;
    story = await fetchStory(words.map(label), PARTS).finally(() => (writing = false));
    if (!story.length) note = "Kunde inte skriva någon saga. Försök igen.";
  }

  const part = $derived(story[at]);
  const limit = $derived(part ? part.text.split(/\s+/).length * PER_WORD + SLACK : 0);
  const clipsOf = (sign: string) => lexicon?.signs.find((each) => each.sign === sign)?.references ?? [];
  const sentence: Sign[] = $derived(
    (part?.words ?? [])
      .filter((each) => cards[each])
      .map((each) => ({ sign: cards[each].sign, references: clipsOf(cards[each].sign), spoken: each })),
  );

  /** The part's text cut into what is signed and what is only spoken, so each word can be coloured. */
  const pieces = $derived(part ? cut(part.text, part.words) : []);

  function scored(scoredAttempt: Attempt | null, told_: string) {
    (attempt = scoredAttempt), (note = told_);
    // A recording that could not be located at all is no answer: the story waits for another one.
    // A word that was not said is not that case — the server scores it as a miss of its sign.
    if (!scoredAttempt?.signs.length) return void recorder?.arm(true);
    for (const judged of scoredAttempt.signs) {
      const spoken = part.words.find((each) => cards[each]?.sign === judged.sign) ?? word(judged.sign);
      const correct = !!judged.correct && judged.usable;
      verdicts = { ...verdicts, [same(spoken)]: correct };
      told = [...told, { word: spoken, correct }];
      if (cards[spoken]) progress = record(progress, cards[spoken], correct);
    }
    save(progress);
    if (waiting) clearTimeout(waiting); // a second recording of the same part replaces the first
    waiting = setTimeout(next, SHOWN); // the colours are the point of the pause, not the verdict
  }

  function next() {
    (at += 1), (verdicts = {}), (attempt = null), (note = "");
  }

  const done = $derived(story.length > 0 && at >= story.length);
  const right = $derived(told.filter((each) => each.correct).length);
</script>

{#if !lexicon}
  <section class="card">
    <h1>Sagan</h1>
    <p class="dim">{note || "Laddar lexikonet …"}</p>
  </section>
{:else if !story.length}
  <section class="card">
    <h1>Sagan</h1>
    <p class="dim">
      En saga skrivs av orden du redan kan, med tyngdpunkt på dem som sitter sämst. Du läser den högt,
      en del i taget, och tecknar orden som är markerade. Sagan fortsätter vad som än händer.
    </p>
    <p class="dim">{pool.length} tecken att välja ur.</p>
    <button onclick={begin} disabled={writing || pool.length < 2}>
      {writing ? "Skriver sagan …" : "Skriv sagan"}
    </button>
    {#if pool.length < 2}
      <p class="dim">Lär dig några <a href="/träna/nya">nya ord</a> först — sagan skrivs av det du kan.</p>
    {/if}
    {#if note}<p class="dim">{note}</p>{/if}
  </section>
{:else if done}
  <section class="card">
    <h1>Sagan är slut</h1>
    <p>{right} av {told.length} tecken rätt.</p>
    <ul class="told">
      {#each told as each, index (index)}
        <li class:ok={each.correct} class:bad={!each.correct}>{each.correct ? "✓" : "✗"} {each.word}</li>
      {/each}
    </ul>
    <button onclick={begin}>En saga till</button>
  </section>
{:else}
  <section class="card">
    <h1>Del {at + 1} av {story.length}</h1>
    <p class="story">
      {#each pieces as piece, index (index)}
        {#if piece.key}
          <span class="key" class:ok={verdicts[same(piece.text)] === true} class:bad={verdicts[same(piece.text)] === false}>
            {piece.text}
          </span>
        {:else}{piece.text}{/if}
      {/each}
    </p>
    <p class="dim">Läs högt och teckna de markerade orden.</p>
    {#if note}<p class="dim">{note}</p>{/if}
  </section>
  <Recorder bind:this={recorder} {sentence} {lexicon} {limit} onattempt={scored} unheardIsMiss />
{/if}

<style>
  .story {
    font-size: 22px;
    line-height: 1.5;
  }

  .key {
    color: var(--accent);
    font-weight: 600;
  }

  .key.ok {
    color: var(--ok);
  }

  .key.bad {
    color: var(--bad);
  }

  .told {
    margin: 8px 0;
    padding-left: 20px;
  }
</style>
