<script lang="ts">
  import { referenceUrl, word, type Sign } from "$lib/api";

  let { sentence, onremove, onclear }: { sentence: Sign[]; onremove: (index: number) => void; onclear: () => void } = $props();
</script>

<section class="card">
  <h2>{sentence.map((sign) => word(sign.sign)).join(" ")}</h2>
  <p class="dim">
    Varje sökträff du väljer lägger till ett tecken; flera blir en mening. Titta på klippen och spela
    sedan in dig själv när du tecknar dem i ordning. Säg meningen högt medan du tecknar den, för TAKK
    talas och tecknas samtidigt: tecknen hittas utifrån när du säger deras ord. Inspelningen börjar när
    du börjar tala och slutar när du tystnar, så du behöver inte trycka på något. Tryck på mellanslag
    för att börja om.
  </p>
  <button class="secondary" onclick={onclear}>Rensa</button>
  <div class="videos">
    {#each sentence as sign, i (i)}
      <figure>
        <figcaption>
          {i + 1}. {word(sign.sign)}
          <button class="secondary" onclick={() => onremove(i)}>Ta bort</button>
        </figcaption>
        {#each sign.references as clip (clip)}
          <!-- svelte-ignore a11y_media_has_caption -->
          <video src={referenceUrl(clip)} autoplay loop muted playsinline controls></video>
        {/each}
      </figure>
    {/each}
  </div>
</section>
