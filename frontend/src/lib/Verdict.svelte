<script lang="ts">
  import { entry, word, type Attempt, type Judgement } from "$lib/api";

  // `labels` gives the word the learner was asked to sign, which is not always what the sign is
  // called: signs of one form are one class named after its lowest entry, so "grön" is scored as
  // `sts:land-00416`. Without it a sign is called what the lexicon calls it, as in free practice.
  let { attempt, note, labels = {} }: { attempt: Attempt | null; note: string; labels?: Record<string, string> } =
    $props();

  const shown = (sign: string) => labels[sign] ?? word(sign);

  const correct = $derived(attempt ? attempt.signs.filter((sign) => sign.correct).length : 0);

  const headline = $derived.by(() => {
    if (!attempt) return note;
    if (!attempt.signs.length) return attempt.note;
    if (attempt.signs.length > 1) return `${correct} av ${attempt.signs.length} tecken kändes igen.`;
    const only = attempt.signs[0];
    return only.correct ? `Rätt: det var ${shown(only.sign)}.` : `Kändes inte igen som ${shown(only.sign)}.`;
  });

  const good = $derived(!!attempt && attempt.signs.length > 0 && correct === attempt.signs.length);

  function line(sign: Judgement): string {
    if (!sign.usable) return `${shown(sign.sign)}: ${sign.note}`;
    const closest =
      sign.closest!.sign === sign.sign
        ? "också det närmaste tecknet i lexikonet"
        : `närmaste tecken i lexikonet ${word(sign.closest!.sign)} (${entry(sign.closest!.sign)}), poäng ${sign.closest!.score.toFixed(2)}`;
    const note = sign.note === "Det ser bra ut." ? "" : sign.note;
    return `${sign.correct ? "✓" : "✗"} ${shown(sign.sign)}: poäng ${sign.score!.toFixed(2)}, ${closest}. ${note}`;
  }

  const found = $derived(attempt?.split === "speech" ? " Tecknen hittades utifrån orden du sa." : "");
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
