<script lang="ts">
  import "../app.css";

  let { children } = $props();
</script>

<main>
  <nav>
    <a href="/pass">Dagens pass</a>
    <a href="/">Fri övning</a>
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
