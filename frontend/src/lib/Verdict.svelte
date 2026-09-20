<script lang="ts">
  import { entry, word, type Attempt, type Judgement } from "$lib/api";

  let { attempt, note }: { attempt: Attempt | null; note: string } = $props();

  const correct = $derived(attempt ? attempt.signs.filter((sign) => sign.correct).length : 0);

  const headline = $derived.by(() => {
    if (!attempt) return note;
    if (!attempt.signs.length) return attempt.note;
    if (attempt.signs.length > 1) return `${correct} of ${attempt.signs.length} signs recognized.`;
    const only = attempt.signs[0];
    return only.correct ? `Correct: that was ${word(only.sign)}.` : `Not recognized as ${word(only.sign)}.`;
  });

  const good = $derived(!!attempt && attempt.signs.length > 0 && correct === attempt.signs.length);

  function line(sign: Judgement): string {
    if (!sign.usable) return `${word(sign.sign)}: ${sign.note}`;
    const closest =
      sign.closest!.sign === sign.sign
        ? "also the closest sign in the lexicon"
        : `closest sign in the lexicon ${word(sign.closest!.sign)} (${entry(sign.closest!.sign)}), score ${sign.closest!.score.toFixed(2)}`;
    const note = sign.note === "Looks good." ? "" : sign.note;
    return `${sign.correct ? "✓" : "✗"} ${word(sign.sign)}: score ${sign.score!.toFixed(2)}, ${closest}. ${note}`;
  }

  const found = $derived(
    attempt?.split === "speech"
      ? " The signs were found by the words you spoke."
      : attempt?.split === "rests"
        ? " The signs were found by the rests between them."
        : "",
  );
</script>

{#if headline}
  <section class="card">
    <p class="verdict" class:ok={good} class:bad={!good}>{headline}</p>
    {#if attempt?.signs.length}
      <ul>
        {#each attempt.signs as sign, i (i)}
          <li class:ok={sign.correct} class:bad={!sign.correct}>{line(sign)}</li>
        {/each}
      </ul>
      <p class="dim">A score of at least {attempt.threshold.toFixed(2)} counts as the sign.{found}</p>
    {/if}
  </section>
{/if}

<style>
  .verdict {
    font-size: 22px;
    font-weight: 600;
  }

  ul {
    margin: 8px 0;
    padding-left: 20px;
  }
</style>
