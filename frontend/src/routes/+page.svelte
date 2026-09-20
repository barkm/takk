<script lang="ts">
  import Recorder from "$lib/Recorder.svelte";
  import Sentence from "$lib/Sentence.svelte";
  import SignSearch from "$lib/SignSearch.svelte";
  import Verdict from "$lib/Verdict.svelte";
  import { fetchLexicon, type Attempt, type Lexicon, type Sign } from "$lib/api";

  let lexicon: Lexicon | null = $state(null);
  let sentence: Sign[] = $state([]);
  let attempt: Attempt | null = $state(null);
  let note = $state("");

  $effect(() => {
    fetchLexicon()
      .then((loaded) => (lexicon = loaded))
      .catch(() => (note = "The server does not answer. Start it with uv run takk."));
  });

  function pick(sign: Sign) {
    sentence = [...sentence, sign];
    (attempt = null), (note = "");
  }

  function remove(index: number) {
    sentence = sentence.filter((_, i) => i !== index);
    (attempt = null), (note = "");
  }
</script>

{#if lexicon}
  <SignSearch signs={lexicon.signs} onpick={pick} />
  {#if sentence.length}
    <Sentence {sentence} onremove={remove} onclear={() => ((sentence = []), (attempt = null), (note = ""))} />
    <Recorder
      {sentence}
      {lexicon}
      onattempt={(scored, said) => ((attempt = scored), (note = said))}
    />
  {/if}
  <Verdict {attempt} {note} />
{:else}
  <section class="card">
    <h1>Practice a sign or a sentence</h1>
    <p class="dim">{note || "Loading the lexicon …"}</p>
  </section>
{/if}
