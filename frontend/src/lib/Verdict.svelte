<script lang="ts">
  import type { Attempt } from "$lib/api";

  // What was wrong with a recording, and nothing else (user, 2026-09-24). Right and wrong are marked
  // on the word itself, with a tick or a cross beside it, so neither is written here: naming the
  // sign that was signed instead said nothing the learner could act on. What is left is the cases
  // where there is no verdict at all — a recording that could not be used, and the recorder's own
  // message when it has one.
  let { attempt, note }: { attempt: Attempt | null; note: string } = $props();

  // A card is one sign: a story colours its own words and shows no verdict of its own.
  const only = $derived(attempt?.signs[0]);

  const headline = $derived.by(() => {
    if (!attempt) return note;
    if (!only) return attempt.note; // the recording could not be split into the sign at all
    return only.usable ? "" : only.note;
  });
</script>

{#if headline}
  <p class="verdict">{headline}</p>
{/if}

<style>
  /* One line, set apart by a rule: it is read at a glance and then gone. */
  .verdict {
    padding: 12px 16px;
    border-left: 3px solid var(--line);
    border-radius: 0;
    background: var(--surface);
    font-size: 17px;
    font-weight: 500;
  }
</style>
