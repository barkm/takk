<script lang="ts">
  import { base } from "$app/paths";
import { page } from "$app/state";

  import { fetchLexicon } from "$lib/api";
  import { Camera, provideCamera } from "$lib/camera.svelte";

  import "../app.css";

  let { children } = $props();

  // The three sections are always one click away (user, 2026-09-23): a rail beside the page on a
  // desktop window, the same three items as a bar along the bottom on a phone. There is no menu page
  // and no way back, since every page is reachable from every other one.
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
  <nav class="rail">
    <span class="brand">Takk</span>
    {#each sections as section (section.href)}
      <a href="{base}{section.href}" class:on={path.startsWith(section.at)}>{section.label}</a>
    {/each}
  </nav>
  <main>{@render children()}</main>
</div>

<style>
  .app {
    display: grid;
    grid-template-columns: var(--rail) 1fr;
    min-height: 100dvh;
  }

  .rail {
    position: sticky;
    top: 0;
    align-self: start;
    height: 100dvh;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 24px 12px;
    background: var(--surface);
    border-right: 1px solid var(--line);
  }

  .brand {
    margin: 4px 0 20px;
    padding: 0 12px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--dim);
  }

  .rail a {
    padding: 10px 12px;
    border-radius: 10px;
    color: var(--dim);
    font-weight: 500;
    text-decoration: none;
  }

  .rail a:hover {
    color: var(--text);
  }

  .rail a.on {
    color: var(--text);
    background: var(--raised);
  }

  main {
    width: 100%;
    max-width: 1080px;
    margin: 0 auto;
    padding: 40px 32px 64px;
  }

  /* On a phone the rail becomes the bar along the bottom, where a thumb reaches it. */
  @media (max-width: 720px) {
    .app {
      grid-template-columns: 1fr;
    }

    .rail {
      position: fixed;
      inset: auto 0 0;
      height: auto;
      z-index: 2;
      flex-direction: row;
      justify-content: space-around;
      padding: 6px;
      padding-bottom: max(6px, env(safe-area-inset-bottom));
      border-right: none;
      border-top: 1px solid var(--line);
    }

    .brand {
      display: none;
    }

    main {
      padding: 20px 16px 88px;
    }
  }
</style>
