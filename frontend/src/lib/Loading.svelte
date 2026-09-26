<script lang="ts">
  // The one loader of the app (user, 2026-09-26): a thin line in the accent with a stretch of it
  // sliding across, square like everything else. It stands wherever something is being waited for —
  // the camera starting, a search, the lexicon — in place of a line of text saying so.
  let { label = "Laddar" }: { label?: string } = $props();
</script>

<div class="loading" role="progressbar" aria-label={label}></div>

<style>
  .loading {
    position: relative;
    height: 2px;
    overflow: hidden;
    background: color-mix(in srgb, var(--accent) 15%, transparent);
  }

  .loading::after {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 30%;
    background: var(--accent);
    animation: slide 1.1s ease-in-out infinite;
  }

  @keyframes slide {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(340%);
    }
  }

  /* where motion is asked to be kept down the stretch stands still and only breathes */
  @media (prefers-reduced-motion: reduce) {
    .loading::after {
      width: 100%;
      animation: breathe 1.6s ease-in-out infinite alternate;
    }

    @keyframes breathe {
      from {
        opacity: 0.2;
      }
    }
  }
</style>
