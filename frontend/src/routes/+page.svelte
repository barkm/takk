<script lang="ts">
  import { goto } from "$app/navigation";

  // There is no menu any more (user, 2026-09-23): the rail is the menu, and the app opens on the
  // practice it was opened to do. The hand is asked for under Träna, where the first recording is.
  $effect(() => void goto("/träna/ord", { replaceState: true }));
</script>
