<script lang="ts">
  import { hand, setHand, type Hand } from "$lib/hand";

  // Menyn (step 8 of ROADMAP-takk.md): the app is three sections and nothing else. The first time it
  // is opened it asks which hand the learner signs with, since every recording is mirrored by it and
  // asking once here keeps the switch off the camera screens.
  let signs = $state<Hand | null>(null);
  let asked = $state(false);

  $effect(() => {
    (signs = hand()), (asked = true);
  });

  function choose(which: Hand) {
    setHand(which);
    signs = which;
  }
</script>

{#if asked && !signs}
  <p>Vilken hand tecknar du med?</p>
  <nav class="choices">
    <button onclick={() => choose("right")}>Höger</button>
    <button onclick={() => choose("left")}>Vänster</button>
  </nav>
{:else}
  <nav class="choices">
    <a href="/sök">Sök</a>
    <a href="/träna">Träna</a>
    <a href="/tecken">Tecken</a>
  </nav>
{/if}
