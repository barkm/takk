<script lang="ts" generics="T extends string | number">
  // One option on a start card: its name and the values it offers, the chosen one marked. The
  // options are set before a pass starts and never while one is running (user, 2026-09-24).
  let {
    label,
    values,
    value = $bindable(),
  }: { label: string; values: T[]; value: T } = $props();
</script>

<div class="choice">
  <span class="dim">{label}</span>
  <div class="values">
    {#each values as each (each)}
      <button class:on={each === value} aria-pressed={each === value} onclick={() => (value = each)}>
        {each}
      </button>
    {/each}
  </div>
</div>

<style>
  .choice {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }

  .values {
    display: flex;
    gap: 2px;
    padding: 3px;
    background: var(--bg);
    border: 1px solid var(--line);
    border-radius: 10px;
  }

  .values button {
    min-width: 44px;
    padding: 6px 0;
    border-radius: 7px;
    background: transparent;
    color: var(--dim);
    font-weight: 500;
  }

  .values button:hover:not(.on) {
    color: var(--text);
    filter: none;
  }

  .values button.on {
    background: var(--raised);
    color: var(--text);
  }
</style>
