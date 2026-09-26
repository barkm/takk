<script lang="ts">
  import { goto } from "$app/navigation";
  import { base } from "$app/paths";
  import { about, setAbout, type Level } from "$lib/about";
  import { hand, setHand, type Hand } from "$lib/hand";

  // Om dig (step 23 of ROADMAP-takk.md): the page the app opens on the first time, and a section of
  // its own after that. Three questions in the learner's words and nothing about what they are for:
  // the hand every recording is mirrored by, whether they have signed before, and who they sign with
  // and where, which is what the chips on Sök and the text under Meningar are fitted to.
  let signs = $state<Hand | null>(null);
  let level = $state<Level | null>(null);
  let context = $state("");
  let first = $state(false); // nothing was said yet, so saving goes on to Sök

  $effect(() => {
    const said = about();
    (signs = hand()), (level = said?.level ?? null), (context = said?.context ?? ""), (first = !said);
  });

  function save() {
    setHand(signs!);
    setAbout({ level: level!, context: context.trim() });
    if (first) goto(`${base}/sök`);
  }

  const levels: [Level, string][] = [
    ["ny", "Nej"],
    ["lite", "Lite"],
    ["van", "Ja"],
  ];
</script>

<section class="card about">
  <h2>Vilken hand tecknar du med?</h2>
  <div class="row">
    {#each [["right", "Höger"], ["left", "Vänster"]] as [which, name] (which)}
      <button class:secondary={signs !== which} aria-pressed={signs === which} onclick={() => (signs = which as Hand)}>
        {name}
      </button>
    {/each}
  </div>

  <h2>Har du tecknat förut?</h2>
  <div class="row">
    {#each levels as [which, name] (which)}
      <button class:secondary={level !== which} aria-pressed={level === which} onclick={() => (level = which)}>
        {name}
      </button>
    {/each}
  </div>

  <h2>Vem tecknar du med, och var?</h2>
  <textarea rows="3" placeholder="Till exempel: förskollärare, barn 3–5 år" bind:value={context}></textarea>

  <div class="row">
    <button disabled={!signs || !level} onclick={save}>{first ? "Börja" : "Spara"}</button>
  </div>
</section>

<style>
  .about {
    max-width: 480px;
  }

  .about h2 {
    font-size: 18px;
    margin: 12px 0 0;
  }

  textarea {
    resize: vertical;
  }
</style>
