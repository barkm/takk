<script lang="ts">
  import { base } from "$app/paths";
import { page } from "$app/state";

  import { fetchLexicon } from "$lib/api";
  import { Camera, provideCamera } from "$lib/camera.svelte";

  import "../app.css";

  let { children } = $props();

  // The three sections are always one click away (user, 2026-09-23): a line above the page on a
  // desktop window (user, 2026-09-26), the same three items as a bar along the bottom on a phone.
  // There is no menu page and no way back, since every page is reachable from every other one.
  const sections = [
    { href: "/sök", label: "Sök", at: "/sök" },
    { href: "/träna/ord", label: "Träna", at: "/träna" },
    { href: "/tecken", label: "Tecken", at: "/tecken" },
  ];

  // The camera itself is mounted by the pages that use it (user, 2026-09-23), so Sök's results and
  // Tecken cost nothing; what the app owns is the lexicon and the handle the pages reach it through,
  // which is this context.
  const camera = new Camera();
  provideCamera(camera);

  $effect(() => {
    fetchLexicon()
      .then((loaded) => {
        camera.lexicon = loaded; // in a block: a `$state` object reads back as its proxy, not as the value assigned
      })
      // The page is deployed apart from the API and reaches a learner who cannot start it, so this
      // says what is wrong and nothing about how to fix it (see step 18 of ROADMAP-takk.md).
      .catch(() => (camera.fit = "Servern svarar inte just nu."));
  });

  // The path carries the sections' Swedish names percent-encoded, which no link here is written in.
  const path = $derived(decodeURIComponent(page.url.pathname).slice(base.length));
</script>

<div class="app">
  <nav class="bar">
    {#each sections as section (section.href)}
      <a href="{base}{section.href}" class:on={path.startsWith(section.at)}>{section.label}</a>
    {/each}
  </nav>
  <main>{@render children()}</main>
</div>

<style>
  /* The bar and the page share one narrow column down the middle of the window. */
  .bar,
  main {
    width: 100%;
    max-width: var(--column);
    margin: 0 auto;
  }

  .bar {
    display: flex;
    justify-content: center;
    gap: 28px;
    padding: 28px 24px 0;
  }

  .bar a {
    padding: 4px 0;
    border-bottom: 2px solid transparent;
    color: var(--dim);
    font-weight: 500;
  }

  .bar a:hover {
    color: var(--text);
    text-decoration: none;
  }

  .bar a.on {
    color: var(--text);
    border-bottom-color: var(--accent);
  }

  main {
    padding: 40px 24px 64px;
  }

  /* On a phone the bar moves along the bottom, where a thumb reaches it. */
  @media (max-width: 720px) {
    .bar {
      position: fixed;
      inset: auto 0 0;
      z-index: 2;
      max-width: none;
      justify-content: space-around;
      padding: 10px 6px;
      padding-bottom: max(10px, env(safe-area-inset-bottom));
      background: var(--bg);
      border-top: 1px solid var(--line);
    }

    main {
      padding: 20px 16px 88px;
    }
  }
</style>
