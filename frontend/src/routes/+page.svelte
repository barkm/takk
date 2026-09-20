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
      .catch(() => (note = "Servern svarar inte. Starta den med uv run takk."));
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
    <h1>Öva ett tecken eller en mening</h1>
    <p class="dim">{note || "Laddar lexikonet …"}</p>
  </section>
{/if}
