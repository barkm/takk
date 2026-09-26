<script lang="ts">
  import Recorder from "$lib/Recorder.svelte";
  import { useCamera } from "$lib/camera.svelte";
  import { fetchStory, word, type Attempt, type SignWord, type Sign, type StoryPart } from "$lib/api";
  import { known, load, record, save, type Progress } from "$lib/progress";
  import { pieces as cut } from "$lib/text";

  // Meningar (step 13 of ROADMAP-takk.md): a connected Swedish text over the words the learner
  // already knows, read line by line. The text scrolls (user, 2026-09-24): the line to sign now is
  // solid and held in the middle, the lines before and after it grey, and it goes on whatever the
  // verdict was — nothing is pressed between the lines. To the learner this is simply how words are
  // repeated, so the page names it nothing and explains nothing.
  // Only words already practised are used, drawn towards the low boxes, so it leans on the weak ones.
  //
  // A story is one text and it ends (user, 2026-09-24). It is written the moment the component is
  // shown — the Träna page is its start card — and signing its last line hands the page back to that
  // card, where the next story is asked for: a story written
  // under the finished one would be a second story with its own beginning, which reads as a break in
  // the text now that the whole of it stands on the page. It is always a long one, of about ten parts:
  // the learner is not asked for a length (user, 2026-09-25).
  //
  // How long a line may be recorded for: it is read aloud, so its length is its text and not its
  // signs — a wordy line with one sign would be cut off by the one sign's worth of seconds a card
  // gets. Swedish read aloud runs at about two words a second; the slack is for reading it at all.
  const PER_WORD = 0.7;
  const SLACK = 4;
  const PARTS = 10; // how many parts a story is asked for; the writer may answer with one more or fewer
  const SHOWN = 1800;
  // the placeholder lines: the width of the words before a word to sign, of that word, and after it
  const SHAPES = [
    [30, 14, 38],
    [22, 16, 30],
    [36, 12, 26],
    [18, 14, 40],
    [26, 12, 34],
    [40, 16, 18],
    [20, 14, 36],
    [32, 12, 28],
  ]; // ms the coloured words stay up before the next line

  let { ondone }: { ondone: (note?: string) => void } = $props();

  const camera = useCamera(); // the camera of the Träna layout, which both modes share
  const lexicon = $derived(camera.lexicon);
  let progress: Progress = $state(load());
  let story: StoryPart[] = $state([]);
  let cards: Record<string, SignWord> = $state({}); // the card behind each word of the story, by word
  let at = $state(0); // the line being signed
  let writing = $state(true);
  let attempt: Attempt | null = $state(null);
  let note = $state("");
  let verdicts: Record<number, Record<number, boolean>> = $state({}); // per line, by sign of the line
  let waiting: ReturnType<typeof setTimeout> | null = null; // the pause the coloured words are read in
  let recorder: ReturnType<typeof Recorder> | undefined = $state();

  const label = (each: SignWord) => each.word ?? word(each.sign);

  /** Write a story over the words this learner knows, weakest first. Twice as many words are
   * offered as there are lines, so the model has something to choose from in every one of them. */
  async function write() {
    const words = known(progress, PARTS * 2);
    cards = Object.fromEntries(words.map((each) => [label(each), each]));
    const written = await fetchStory(words.map(label), PARTS).finally(() => (writing = false));
    if (!written.length) return void ondone("Kunde inte skriva någon text. Försök igen.");
    story = written;
  }

  write();

  const line = $derived(story[at]);
  const limit = $derived(line ? line.text.split(/\s+/).length * PER_WORD + SLACK : 0);
  const clipsOf = (sign: string) => lexicon?.signs.find((each) => each.sign === sign)?.references ?? [];
  // One entry per sign of the line, in the order they are spoken, which is the order the server
  // answers in: the verdicts come back as a list parallel to this one. The form is what is spoken,
  // so it is what the alignment times; the word is what names the sign. Every word of a line was
  // offered from `cards`, which is what the story was written over, so the card is always there.
  const sentence: Sign[] = $derived(
    (line?.signs ?? []).map(({ said, word: each }) => ({
      sign: cards[each].sign,
      references: clipsOf(cards[each].sign),
      spoken: said,
    })),
  );

  /** The whole story so far, each line cut into what is signed and what is only spoken. It is read
   * as one text that scrolls (user, 2026-09-24): the lines already signed stay above with the
   * colours they were given, and the line being signed is scrolled to the middle as it comes up. */
  const shown = $derived(
    story.map((each, index) => ({ index, pieces: cut(each.text, each.signs.map((sign) => sign.said)) })),
  );

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
    // The verdicts come back in the order the signs were sent, so they are read by position rather
    // than looked up by the sign they scored: a line may sign one word twice, and two words of the
    // learner's own can be one sign form ("blå" and "öga"), so a sign does not name its verdict.
    const judged = scoredAttempt.signs.map((each) => !!each.correct && each.usable);
    verdicts = { ...verdicts, [at]: { ...verdicts[at], ...judged } };
    // A word signed twice in one line is still one answer, so the box moves once per line: it is
    // correct only when every occurrence of it was.
    const answers = new Map<string, boolean>();
    line.signs.forEach(({ word: each }, index) => {
      answers.set(each, (answers.get(each) ?? true) && judged[index]);
    });
    for (const [each, correct] of answers) progress = record(progress, cards[each], correct);
    save(progress);
    if (waiting) clearTimeout(waiting); // a second recording of the same line replaces the first
    waiting = setTimeout(next, SHOWN); // the colours are the point of the pause, not the verdict
  }

  function next() {
    (attempt = null), (note = "");
    // The last line signed ends the story, and the page is the start card again.
    if (at + 1 < story.length) return void (at += 1);
    ondone();
  }
</script>

{#if writing}
  <!-- The lines' shape while the story is written, which takes seconds, standing where the text will
       (user, 2026-09-26): in the same scrolling box, the first line solid and the rest faded as the
       lines around the one to sign are. Each line has a stretch of the accent in it, where a word to
       sign will be. -->
  <div class="story" aria-label="Skriver" aria-busy="true">
    {#each SHAPES as shape, index (index)}
      <p class:now={index === 0} class="shape">
        <span class="skeleton" style="width: {shape[0]}%"></span>
        <span class="skeleton key" style="width: {shape[1]}%"></span>
        <span class="skeleton" style="width: {shape[2]}%"></span>
      </p>
    {/each}
  </div>
{:else}
  <div class="story">
    {#each shown as { index, pieces } (index)}
      <p bind:this={lines[index]} class:now={index === at}>
        {#each pieces as piece, k (k)}
          {#if piece.at !== null}
            <span
              class="key"
              class:ok={piece.at !== null && verdicts[index]?.[piece.at] === true}
              class:bad={piece.at !== null && verdicts[index]?.[piece.at] === false}>{piece.text}</span>
          {:else}{piece.text}{/if}
        {/each}
      </p>
    {/each}
  </div>
  {#if note}<p class="dim">{note}</p>{/if}
  <Recorder bind:this={recorder} {sentence} {limit} onattempt={scored} unheardIsMiss />
{/if}

<style>
  /* The text is the page: the line to sign is set large and solid, the lines around it recede. The
     whole story scrolls inside this box, which the line being signed is kept in the middle of; the
     padding is what lets the first and the last line reach that middle, and the mask fades the text
     out at both ends rather than cutting it off at an edge. */
  .shape {
    display: flex;
    gap: 10px;
  }

  /* darker than a skeleton tile, since these stand faded like the lines around the one to sign */
  .shape span {
    display: block;
    height: 0.9em;
    margin: 0.28em 0;
    background: #d9dce2;
  }

  .shape .key {
    background: var(--accent);
  }

  .story {
    animation: appear 0.3s ease;
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

  /* A word to sign is set in the accent (user, 2026-09-26): the vivid blue reads as text and is far
     from the verdicts' green and red. A highlight behind the word was tried while the accent was a
     faint sage, and dropped with it. */
  .key {
    color: var(--accent);
    font-weight: 600;
    transition: color 0.25s ease; /* the verdict colours the word in rather than switching it */
  }

  @keyframes appear {
    from {
      opacity: 0;
    }
  }

  .key.ok {
    color: var(--ok);
  }

  .key.bad {
    color: var(--bad);
  }
</style>
