<script lang="ts">
  import Recorder from "$lib/Recorder.svelte";
  import { useCamera } from "$lib/camera.svelte";
  import { fetchStory, word, type Attempt, type SignWord, type Sign, type StoryPart } from "$lib/api";
  import { known, load, record, save, type Progress } from "$lib/progress";
  import { pieces as cut } from "$lib/text";

  // Meningar (step 13 of ROADMAP-takk.md): a connected Swedish text over the words the learner
  // already knows, read line by line. The text scrolls (user, 2026-09-24): the line to sign now is
  // solid and held in the middle, the lines before and after it grey, and it goes on whatever the
  // verdict was — nothing is pressed and there is no end screen. To the learner
  // this is simply how words are repeated, so the page names it nothing and explains nothing.
  // Only words already practised are used, drawn towards the low boxes, so it leans on the weak ones.
  const LINES = 6; // lines written per request; the next ones are asked for as the learner reaches them
  // How long a line may be recorded for: it is read aloud, so its length is its text and not its
  // signs — a wordy line with one sign would be cut off by the one sign's worth of seconds a card
  // gets. Swedish read aloud runs at about two words a second; the slack is for reading it at all.
  const PER_WORD = 0.7;
  const SLACK = 4;
  const SHOWN = 1800; // ms the coloured words stay up before the next line

  const camera = useCamera(); // the camera of the Träna layout, which both modes share
  const lexicon = $derived(camera.lexicon);
  let progress: Progress = $state({});
  let story: StoryPart[] = $state([]);
  let cards: Record<string, SignWord> = $state({}); // the card behind each word of the story, by word
  let at = $state(0); // the line being signed
  let writing = $state(false);
  let attempt: Attempt | null = $state(null);
  let note = $state("");
  let verdicts: Record<number, Record<string, boolean>> = $state({}); // per line, by lowercased word
  let waiting: ReturnType<typeof setTimeout> | null = null; // the pause the coloured words are read in
  let stalled = $state(false); // the writer failed, so nothing is asked for again until it is asked for
  let recorder: ReturnType<typeof Recorder> | undefined = $state();

  $effect(() => {
    progress = load();
  });

  const label = (each: SignWord) => each.word ?? word(each.sign);
  // A word is written as the story has it, so "Mamma" opening a line is the same word as "mamma" in
  // the list: verdicts are kept under the lowercase word.
  const same = (text: string) => text.toLowerCase();
  const pool = $derived(known(progress, Infinity, () => 0)); // everything practised, for the count

  /** Write the next lines over the words this learner knows, weakest first. Twice as many words are
   * offered as there are lines, so the model has something to choose from in every one of them. */
  async function write() {
    const words = known(progress, LINES * 2);
    cards = { ...cards, ...Object.fromEntries(words.map((each) => [label(each), each])) };
    writing = true;
    const written = await fetchStory(words.map(label), LINES).finally(() => (writing = false));
    stalled = !written.length;
    if (stalled) return void (note = "Kunde inte skriva någon text. Försök igen.");
    (story = [...story, ...written]), (note = "");
  }

  // The story never ends: when the learner reaches the last written line, the next ones are asked
  // for. A writer that failed is not asked again by itself, or a dead server would be asked forever.
  $effect(() => {
    if (story.length && at >= story.length - 1 && !writing && !stalled) void write();
  });

  const line = $derived(story[at]);
  const limit = $derived(line ? line.text.split(/\s+/).length * PER_WORD + SLACK : 0);
  const clipsOf = (sign: string) => lexicon?.signs.find((each) => each.sign === sign)?.references ?? [];
  const sentence: Sign[] = $derived(
    (line?.words ?? [])
      .filter((each) => cards[each])
      .map((each) => ({ sign: cards[each].sign, references: clipsOf(cards[each].sign), spoken: each })),
  );

  /** The whole story so far, each line cut into what is signed and what is only spoken. It is read
   * as one text that scrolls (user, 2026-09-24): the lines already signed stay above with the
   * colours they were given, and the line being signed is scrolled to the middle as it comes up. */
  const shown = $derived(story.map((each, index) => ({ index, pieces: cut(each.text, each.words) })));

  let lines: HTMLParagraphElement[] = $state([]);

  // The line being signed is scrolled to the middle of the text, whenever it changes and as soon as
  // it is on the page: a line written after this ran lands in `lines`, which the effect reads. The
  // learner is reading rather than scrolling, so a browser told to keep motion down jumps instead.
  $effect(() => {
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    lines[at]?.scrollIntoView({ block: "center", behavior: still ? "auto" : "smooth" });
  });

  function scored(scoredAttempt: Attempt | null, told: string) {
    (attempt = scoredAttempt), (note = told);
    // A recording that could not be located at all is no answer: the story waits for another one.
    // A word that was not said is not that case — the server scores it as a miss of its sign.
    if (!scoredAttempt?.signs.length) return void recorder?.arm(true);
    for (const judged of scoredAttempt.signs) {
      const spoken = line.words.find((each) => cards[each]?.sign === judged.sign) ?? word(judged.sign);
      const correct = !!judged.correct && judged.usable;
      verdicts = { ...verdicts, [at]: { ...verdicts[at], [same(spoken)]: correct } };
      if (cards[spoken]) progress = record(progress, cards[spoken], correct);
    }
    save(progress);
    if (waiting) clearTimeout(waiting); // a second recording of the same line replaces the first
    waiting = setTimeout(next, SHOWN); // the colours are the point of the pause, not the verdict
  }

  function next() {
    (at += 1), (attempt = null), (note = "");
  }
</script>

{#if !lexicon}
  <p class="dim">{note || "Laddar lexikonet ..."}</p>
{:else if !story.length}
  <section class="card">
    <h1>Meningar</h1>
    <p class="dim">En text över orden du redan kan, en rad i taget. Säg raden högt och teckna orden i den.</p>
    <div class="row">
      <button onclick={write} disabled={writing || pool.length < 2}>{writing ? "Skriver ..." : "Börja"}</button>
      {#if pool.length < 2}
        <span class="dim">Öva några <a href="/träna/ord">ord</a> först.</span>
      {/if}
    </div>
    {#if note}<p class="dim">{note}</p>{/if}
  </section>
{:else}
  <div class="story">
    {#each shown as { index, pieces } (index)}
      <p bind:this={lines[index]} class:now={index === at}>
        {#each pieces as piece, k (k)}
          {#if piece.key}
            <span
              class="key"
              class:ok={verdicts[index]?.[same(piece.text)] === true}
              class:bad={verdicts[index]?.[same(piece.text)] === false}>{piece.text}</span>
          {:else}{piece.text}{/if}
        {/each}
      </p>
    {/each}
  </div>
  {#if note}<p class="dim">{note}</p>{/if}
  {#if stalled}<button onclick={write} disabled={writing}>Fortsätt</button>{/if}
  <Recorder bind:this={recorder} {sentence} {limit} onattempt={scored} unheardIsMiss />
{/if}

<style>
  /* The text is the page: the line to sign is set large and solid, the lines around it recede. The
     whole story scrolls inside this box, which the line being signed is kept in the middle of; the
     padding is what lets the first and the last line reach that middle, and the mask fades the text
     out at both ends rather than cutting it off at an edge. */
  .story {
    max-height: 60vh;
    overflow-y: auto;
    padding: 28vh 0;
    font-size: 26px;
    line-height: 1.45;
    letter-spacing: -0.01em;
    mask-image: linear-gradient(to bottom, transparent, #000 18%, #000 82%, transparent);
    scrollbar-width: thin;
  }

  .story p {
    color: var(--dim);
    opacity: 0.4; /* what was signed and what comes next, readable but plainly not the line to sign */
    padding: 6px 0;
    transition: opacity 0.2s ease;
  }

  .story p.now {
    color: var(--text);
    opacity: 1;
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
</style>
