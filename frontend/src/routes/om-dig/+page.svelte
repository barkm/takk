<script lang="ts">
  import { goto } from "$app/navigation";
  import { base } from "$app/paths";
  import { about, setAbout, type Level } from "$lib/about";
  import { hand, setHand, type Hand } from "$lib/hand";

  // Om dig (step 23 of ROADMAP-takk.md): the page the app opens on the first time, and a section of
  // its own after that. Three questions in the learner's words and nothing about what they are for:
  // the hand every recording is mirrored by, whether they have signed before, and who they sign with
  // and where, which is what the chips on Sök and the text under Meningar are fitted to. Last, whether
  // they speak while signing, which is what tells a recording's signs apart (`Recorder.svelte`).
  let signs = $state<Hand | null>(null);
  let level = $state<Level | null>(null);
  let context = $state("");
  let speaks = $state(true);
  let first = $state(false); // nothing was said yet, so saving goes on to Sök
  let saved = $state(""); // the answers as last saved, so Spara is pressable only when they differ

  const answers = $derived(JSON.stringify([signs, level, context.trim(), speaks]));

  $effect(() => {
    const said = about();
    const kept = { signs: hand(), level: said?.level ?? null, context: said?.context ?? "", speaks: said?.speaks ?? true };
    (signs = kept.signs), (level = kept.level), (context = kept.context), (speaks = kept.speaks), (first = !said);
    saved = JSON.stringify([kept.signs, kept.level, kept.context, kept.speaks]); // not read back, so the effect runs once
  });

  function save() {
    setHand(signs!);
    setAbout({ level: level!, context: context.trim(), speaks });
    saved = answers;
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

  <h2>Pratar du medan du tecknar?</h2>
  <div class="row">
    {#each [[true, "Ja"], [false, "Nej"]] as [which, name] (name)}
      <button class:secondary={speaks !== which} aria-pressed={speaks === which} onclick={() => (speaks = which as boolean)}>
        {name}
      </button>
    {/each}
  </div>

  <div class="row">
    <button disabled={!signs || !level || answers === saved} onclick={save}>{first ? "Börja" : "Spara"}</button>
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
