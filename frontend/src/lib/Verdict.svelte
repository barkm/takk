<script lang="ts">
  import { word, type Attempt } from "$lib/api";

  // A card says only whether the sign was right (user, 2026-09-21). `labels` gives the word the
  // learner was asked to sign, which is not always what the sign is called: signs of one form are one
  // class named after its lowest entry, so "grön" is scored as `sts:land-00416`.
  let {
    attempt,
    note,
    labels = {},
  }: { attempt: Attempt | null; note: string; labels?: Record<string, string> } = $props();

  // A card is one sign: a story colours its own words and shows no verdict of its own.
  const only = $derived(attempt?.signs[0]);

  const headline = $derived.by(() => {
    if (!attempt) return note;
    if (!only) return attempt.note;
    return only.correct ? "Rätt" : `Fel — det såg inte ut som ${labels[only.sign] ?? word(only.sign)}`;
  });
</script>

{#if headline}
  <p class="verdict" class:ok={only?.correct} class:bad={only && !only.correct}>{headline}</p>
{/if}

<style>
  /* One line, marked by a rule in its own colour: the verdict is read at a glance and then gone. */
  .verdict {
    padding: 12px 16px;
    border-left: 3px solid var(--line);
    border-radius: 0 8px 8px 0;
    background: var(--surface);
    font-size: 17px;
    font-weight: 500;
  }

  .verdict.ok {
    border-left-color: var(--ok);
    color: var(--ok);
  }

  .verdict.bad {
    border-left-color: var(--bad);
    color: var(--bad);
  }
</style>
