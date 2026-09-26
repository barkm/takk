<script lang="ts">
  import Recorder from "$lib/Recorder.svelte";
  import Verdict from "$lib/Verdict.svelte";
  import { about } from "$lib/about";
  import { fetchForm, word, type Attempt, type SignWord, type Sign } from "$lib/api";
  import { useCamera } from "$lib/camera.svelte";
  import { pass, REPEATS, WORDS } from "$lib/options";
  import { load, record, save, toLearn, type Progress } from "$lib/progress";

  // Ord (steps 9, 12 and 16 of ROADMAP-takk.md): one sign at a time, with its clip and the lexicon's
  // description of the form on screen while it is signed. It teaches the words the learner picked in
  // Sök and the ones they have missed since — the clip is what a word signed wrong needs, and
  // Meningar never stops to show it. An accepted word goes into the first Leitner box, or keeps the
  // box it had when it was not due; a missed one is simply signed again, with the clip still there.
  //
  // A pass is five words, each signed twice (user, 2026-09-26). It starts the moment the component is
  // shown — the Träna page is its start card — and tells the page when its last word is done.
  // How long the verdict's mark stays on the word. A miss holds twice as long (user, 2026-09-24):
  // the word is about to be signed again, so it is worth reading, while an accepted word is on its
  // way out and only has to be seen.
  const MARKED = 700;
  const MISSED = 2 * MARKED;

  let { ondone }: { ondone: () => void } = $props();

  const camera = useCamera(); // the camera of the Träna layout, which both modes share
  const lexicon = $derived(camera.lexicon);
  let progress: Progress = $state(load());
  // the words left, the current one first: missed words first, then picked and never practised
  let queue: SignWord[] = $state(pass(toLearn(load()).slice(0, WORDS), REPEATS));
  let taken = $state(0); // how many of this session's words are done
  let attempt: Attempt | null = $state(null);
  let note = $state("");
  let form = $state(""); // the lexicon's description of the current sign
  let mark = $state<"ok" | "bad" | null>(null); // the verdict on the word, while the card holds it
  let recorder: ReturnType<typeof Recorder> | undefined = $state();

  /** The word on a card, which is not the name of the sign that scores it when several words share
   * one sign form: "blå" is scored by `sts:öga-02636`, the lowest entry of that form. */
  const label = (each: SignWord) => each.word ?? word(each.sign);

  const current = $derived(queue[0]);
  $effect(() => {
    if (!current) ondone(); // the pass is over, and the page is its start card again
  });
  const shown = $derived(current ? label(current) : "");
  const references = $derived(lexicon?.signs.find((each) => each.sign === current?.sign)?.references ?? []);
  const sentence: Sign[] = $derived(current ? [{ sign: current.sign, references, spoken: shown }] : []);

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
    if (!judged.length || !judged[0].usable) return void recorder?.arm(true); // the line below says why
    // The verdict is a mark on the word and nothing written (user, 2026-09-24): a tick for a sign
    // that was right, a cross for one that was not, held for a moment. The card used to move on the
    // instant it was accepted, so the learner never saw that it had been.
    if (!judged[0].correct) {
      mark = "bad";
      return void setTimeout(() => {
        mark = null;
        recorder?.arm(true); // the same word again, with its clip still on screen
      }, MISSED);
    }
    progress = record(progress, { ...current, word: shown }, true);
    save(progress);
    mark = "ok";
    setTimeout(() => {
      mark = null;
      (queue = queue.slice(1)), (taken += 1), (attempt = null), (note = "");
    }, MARKED);
  }
</script>

{#if current}
  <header class="head">
    <div class="meter" style="--done: {taken / (taken + queue.length)}"></div>
    <p class="dim">{taken} av {taken + queue.length} klara.{about()?.speaks === false ? "" : " Säg ordet högt medan du tecknar det."}</p>
  </header>
  <!-- The slot keeps its place whether or not there is a mark in it, so the word does not move when
       one lands; the mark itself is only there while it is shown, since a mark on its way out would
       otherwise change shape and colour as it went. -->
  <h1 class={mark}>
    {shown}
    <span class="mark">
      {#if mark}
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            d={mark === "bad" ? "M5 5 19 19M19 5 5 19" : "M4 13l5 5L20 6"}
            stroke="currentColor"
            stroke-width="3"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      {/if}
    </span>
  </h1>
  {#if references.length}
    <!-- svelte-ignore a11y_media_has_caption -->
    <video src={references[0]} autoplay loop muted playsinline controls></video>
  {/if}
  {#if form}<p class="form dim">{form}</p>{/if}
  <Recorder bind:this={recorder} {sentence} onattempt={scored} />
  <Verdict {attempt} {note} />
{/if}

<style>
  .head {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  /* How far this session has come, as the one piece of chrome the card carries. */
  .meter {
    height: 3px;
    border-radius: 0;
    background: var(--line);
  }

  .meter::after {
    content: "";
    display: block;
    height: 100%;
    width: calc(var(--done) * 100%);
    border-radius: 0;
    background: var(--accent);
    transition: width 0.3s ease;
  }

  h1 {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  /* The whole verdict (user, 2026-09-24): the word turns green with a tick or red with a cross for a
     moment, and nothing is written. A sentence naming the sign that was signed instead told the
     learner nothing they could act on; the clip beside them is what does. */
  .mark {
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
  }

  .mark svg {
    width: 100%;
    height: 100%;
    fill: none;
    animation: land 0.15s ease;
  }

  /* it fades in and is simply gone again, which is why there is no transition on the way out */
  @keyframes land {
    from {
      opacity: 0;
    }
  }

  h1.ok {
    color: var(--ok);
  }

  h1.bad {
    color: var(--bad);
  }

  /* the lexicon's own shape, held before the clip loads so the page does not jump when it arrives */
  video {
    display: block;
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
    border-radius: var(--radius);
    border: 1px solid var(--line);
  }

  .form {
    font-style: italic;
  }
</style>
