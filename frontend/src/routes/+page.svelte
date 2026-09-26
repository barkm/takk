<script lang="ts">
  import { goto } from "$app/navigation";
  import { base } from "$app/paths";

  // There is no menu any more (user, 2026-09-23): the bar is the menu, and the app opens on the
  // practice it was opened to do. The first visit is sent to Om dig by the layout.
  $effect(() => void goto(`${base}/träna`, { replaceState: true }));
</script>
