<script lang="ts">
  import "../app.css";

  let { children } = $props();
</script>

<main>
  <nav>
    <a href="/">Nya ord</a>
    <a href="/saga">Sagan</a>
    <a href="/framsteg">Mina tecken</a>
  </nav>
  {@render children()}
</main>

<style>
  nav {
    display: flex;
    gap: 16px;
    margin-bottom: 12px;
  }
</style>
