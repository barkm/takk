<script lang="ts">
  import { entry, word, type Attempt, type Judgement } from "$lib/api";

  let { attempt, note }: { attempt: Attempt | null; note: string } = $props();

  const correct = $derived(attempt ? attempt.signs.filter((sign) => sign.correct).length : 0);

  const headline = $derived.by(() => {
    if (!attempt) return note;
    if (!attempt.signs.length) return attempt.note;
    if (attempt.signs.length > 1) return `${correct} av ${attempt.signs.length} tecken kändes igen.`;
    const only = attempt.signs[0];
    return only.correct ? `Rätt: det var ${word(only.sign)}.` : `Kändes inte igen som ${word(only.sign)}.`;
  });

  const good = $derived(!!attempt && attempt.signs.length > 0 && correct === attempt.signs.length);

  function line(sign: Judgement): string {
    if (!sign.usable) return `${word(sign.sign)}: ${sign.note}`;
    const closest =
      sign.closest!.sign === sign.sign
        ? "också det närmaste tecknet i lexikonet"
        : `närmaste tecken i lexikonet ${word(sign.closest!.sign)} (${entry(sign.closest!.sign)}), poäng ${sign.closest!.score.toFixed(2)}`;
    const note = sign.note === "Det ser bra ut." ? "" : sign.note;
    return `${sign.correct ? "✓" : "✗"} ${word(sign.sign)}: poäng ${sign.score!.toFixed(2)}, ${closest}. ${note}`;
  }

  const found = $derived(
    attempt?.split === "speech"
      ? " Tecknen hittades utifrån orden du sa."
      : attempt?.split === "rests"
        ? " Tecknen hittades utifrån pauserna mellan dem."
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
      <p class="dim">En poäng på minst {attempt.threshold.toFixed(2)} räknas som tecknet.{found}</p>
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
