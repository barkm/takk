<script lang="ts">
  import { goto, onNavigate } from "$app/navigation";
  import { base } from "$app/paths";
  import { page } from "$app/state";

  import { about } from "$lib/about";
  import { fetchLexicon } from "$lib/api";
  import { Camera, provideCamera } from "$lib/camera.svelte";

  import "../app.css";

  let { children } = $props();

  // Moving between sections slides the underline from one section to the next (user, 2026-09-26),
  // with the browser's view transitions: the underline is one element with a name of its own, so the
  // browser moves it, while the page itself switches at once (`app.css`). A browser without them, or
  // one asked to keep motion down, simply switches.
  onNavigate((navigation) => {
    if (!document.startViewTransition || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    return new Promise((resolve) => {
      document.startViewTransition(async () => {
        resolve();
        await navigation.complete;
      });
    });
  });

  // The sections are always one click away (user, 2026-09-23): a line above the page on a
  // desktop window (user, 2026-09-26), the same three items as a bar along the bottom on a phone.
  // There is no menu page and no way back, since every page is reachable from every other one.
  const sections = [
    { href: "/sök", label: "Sök", at: "/sök", name: "sök" },
    { href: "/träna", label: "Träna", at: "/träna", name: "träna" },
    { href: "/tecken", label: "Tecken", at: "/tecken", name: "tecken" },
    { href: "/om-dig", label: "Om dig", at: "/om-dig", name: "om-dig" },
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
  const current = $derived(sections.find((section) => path.startsWith(section.at))?.name);

  // The first time the app is opened, whatever page it is opened on, it asks Om dig first (step 23):
  // the hand every recording needs, and who the learner is, which the chips on Sök are made from.
  // No other page is shown until then, so none opens the camera under the question.
  let ready = $state(false);

  $effect(() => {
    ready = !!about() || current === "om-dig";
    if (!ready) void goto(`${base}/om-dig`, { replaceState: true });
  });

</script>

<div class="app">
  <nav class="bar">
    {#each sections as section (section.href)}
      <a href="{base}{section.href}" class:on={current === section.name}>
        {section.label}{#if current === section.name}<span class="line"></span>{/if}
      </a>
    {/each}
  </nav>
  <main>{#if ready}{@render children()}{/if}</main>
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
    position: relative;
    padding: 4px 0 6px;
    color: var(--dim);
    font-weight: 500;
  }

  .bar a:hover {
    color: var(--text);
    text-decoration: none;
  }

  /* the current section underlined in the accent (user, 2026-09-26, after a filled pill): the menu
     stays quiet, and the blue only marks where the learner is */
  .bar a.on {
    color: var(--text);
    font-weight: 600;
  }

  .line {
    position: absolute;
    inset: auto 0 0;
    height: 2px;
    background: var(--accent);
    view-transition-name: section-line;
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
