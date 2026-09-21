<script lang="ts">
  import { fetchLexicon, fetchSearch, referenceUrl, word, type Lexicon, type SignWord } from "$lib/api";
  import { add, load, save, type Progress } from "$lib/progress";

  // Tecken (step 9 of ROADMAP-takk.md): the one way vocabulary grows. A search for a word or a theme
  // lists signs to tick, and what is ticked is what Nya ord teaches. Searching by signing is step 10.
  const WAIT = 200; // milliseconds after the last keystroke, so a word is searched once and not per letter

  let lexicon = $state<Lexicon | null>(null);
  let query = $state("");
  let results: SignWord[] = $state([]);
  let progress: Progress = $state({});
  let note = $state("");
  let timer: ReturnType<typeof setTimeout>;

  $effect(() => {
    progress = load();
    fetchLexicon()
      .then((loaded) => (lexicon = loaded))
      .catch(() => (note = "Servern svarar inte. Starta den med uv run takk."));
  });

  const clips = $derived(new Map(lexicon?.signs.map((each) => [each.sign, each.references]) ?? []));
  const label = (each: SignWord) => each.word ?? word(each.sign);

  function search(text: string) {
    clearTimeout(timer);
    query = text;
    timer = setTimeout(async () => (results = text.trim() ? await fetchSearch(text) : []), WAIT);
  }

  function pick(each: SignWord, on: boolean) {
    // Unticking is only ever undoing the tick: a sign that has been practised keeps its box, so its
    // checkbox stays on and disabled rather than throwing away what is learned of it.
    if (on) {
      progress = add(progress, [each]);
    } else {
      const { [each.sign]: dropped, ...rest } = progress;
      progress = rest;
    }
    save(progress);
  }

  function pickAll() {
    progress = add(progress, results);
    save(progress);
  }
</script>

<input
  type="search"
  placeholder="Sök efter ord eller teman"
  value={query}
  oninput={(event) => search(event.currentTarget.value)}
/>

{#if note}
  <p class="dim">{note}</p>
{/if}

{#if results.length}
  <button onclick={pickAll}>Lägg till alla</button>

  <ul>
    {#each results as each (each.sign)}
      <li>
        <label>
          <input
            type="checkbox"
            checked={!!progress[each.sign]}
            disabled={progress[each.sign]?.box > 0}
            onchange={(event) => pick(each, event.currentTarget.checked)}
          />
          {label(each)}
        </label>
        {#if clips.get(each.sign)?.length}
          <!-- svelte-ignore a11y_media_has_caption -->
          <video src={referenceUrl(clips.get(each.sign)![0])} preload="none" loop muted playsinline controls></video>
        {/if}
      </li>
    {/each}
  </ul>
{/if}

<style>
  button {
    margin: 12px 0;
  }

  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 6px 0;
    border-bottom: 1px solid var(--line);
  }

  video {
    width: 140px;
    border-radius: 4px;
    background: #000;
  }
</style>
