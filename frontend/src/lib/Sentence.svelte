<script lang="ts">
  import { referenceUrl, word, type Sign } from "$lib/api";

  let { sentence, onremove, onclear }: { sentence: Sign[]; onremove: (index: number) => void; onclear: () => void } = $props();
</script>

<section class="card">
  <h2>{sentence.map((sign) => word(sign.sign)).join(" ")}</h2>
  <p class="dim">
    Each search result you pick adds a sign; several make a sentence. Watch the clips, then record
    yourself signing them in order. Say the sentence aloud as you sign it, as TAKK is spoken and signed
    at once: the signs are found by when you say their words. Without a microphone, lower your hands
    between the signs instead. Press space to start and stop recording.
  </p>
  <button class="secondary" onclick={onclear}>Clear</button>
  <div class="videos">
    {#each sentence as sign, i (i)}
      <figure>
        <figcaption>
          {i + 1}. {word(sign.sign)}
          <button class="secondary" onclick={() => onremove(i)}>Remove</button>
        </figcaption>
        {#each sign.references as clip (clip)}
          <!-- svelte-ignore a11y_media_has_caption -->
          <video src={referenceUrl(clip)} autoplay loop muted playsinline controls></video>
        {/each}
      </figure>
    {/each}
  </div>
</section>
