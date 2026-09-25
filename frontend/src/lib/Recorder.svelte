<script lang="ts">
  import { untrack } from "svelte";

  import { scoreAttempt, type Attempt, type Sign } from "$lib/api";
  import { useCamera } from "$lib/camera.svelte";
  import { hand } from "$lib/hand";
  import { hear, listening, type Phase } from "$lib/voice";

  let {
    sentence,
    onattempt,
    unheardIsMiss = false,
    limit,
  }: {
    sentence: Sign[];
    onattempt: (attempt: Attempt | null, note: string) => void;
    /** Story mode: a word that was not said is a miss of its sign rather than a refused recording. */
    unheardIsMiss?: boolean;
    /** How long the recording may run, in seconds. A card is as long as its signs, which is the
     * default; a part of a story is as long as its text, most of which is spoken and not signed. */
    limit?: number;
  } = $props();

  const NO_MICROPHONE = "Mikrofonen behövs: orden du säger högt är det som visar var tecknen är i inspelningen.";

  const TICK = 50; // ms between readings of the microphone; each covers the newest 21 ms of sound
  const IDLE = 10; // seconds armed without a word before the recording is thrown away and started over

  // The camera, the landmarker and the framing feedback belong to the Träna layout, which keeps them
  // up while the modes come and go (step 12 of ROADMAP-takk.md); this component only records.
  const camera = useCamera();
  const tracker = $derived(camera.tracker);
  const lexicon = $derived(camera.lexicon!); // a mode is only shown once the layout has the lexicon
  let phase = $state<Phase | "idle">("idle");
  let scoring = $state(false);
  let seconds = $state(0);
  let ticker: ReturnType<typeof setInterval> | null = null;
  let ears = listening();
  let since = 0; // when the current phase began, to arm afresh and to time the attempt

  // Every attempt is located by the words the signer says, so without the microphone there is
  // nothing to locate it with. Refusing here says so before a recording is made and thrown away.
  const silent = $derived(tracker?.hasAudio === false);

  // A new sentence arms itself, so the learner signs it when they are ready rather than after
  // pressing anything. The camera has to be up first, which it may not be when the card appears.
  $effect(() => {
    if (sentence.length && tracker && !silent) untrack(arm); // arming writes what it reads, so only the sentence triggers it
  });

  /** Record from now, and wait for the signer to speak. The recording runs from here rather than
   * from the first word because the hands rise before the voice; the server keeps the part around
   * the speech and drops the rest (`PRE_ROLL` in speech.py).
   *
   * The last verdict is cleared, since it belongs to the recording being replaced. A caller that arms
   * the same card again after a miss keeps it instead (`keep`), because it is what the learner reads
   * and it is what shows the clip while they sign it over. */
  export async function arm(keep = false) {
    if (!tracker || silent) return;
    if (!keep) onattempt(null, "");
    tracker.resume();
    const again = phase !== "idle";
    phase = "idle"; // the readings stand still while a recording already running is closed and dropped
    if (again) await tracker.stopRecording();
    tracker.startRecording();
    (ears = listening()), (seconds = 0);
    (phase = ears.phase), (since = Date.now());
    if (!ticker) ticker = setInterval(listen, TICK);
  }

  /** One reading of the microphone, every TICK ms. `hear` decides what it means; the clock is this
   * side, since waiting and talking too long are not things a level can tell. */
  function listen() {
    if (!tracker || phase === "idle") return;
    ears = hear(ears, tracker.level);
    if (ears.phase !== phase) (phase = ears.phase), (since = Date.now());
    if (phase === "done") return void stop();
    // Waiting armed keeps recording, so a long wait is thrown away rather than sent and aligned.
    if (phase === "armed" && Date.now() - since > IDLE * 1000) return void arm();
    if (phase !== "speaking") return;
    seconds = (Date.now() - since) / 1000;
    if (seconds > (limit ?? lexicon.maxSeconds * sentence.length)) stop();
  }

  async function stop() {
    if (!tracker || phase === "idle") return; // the silence, the time limit and arming again all stop one
    phase = "idle";
    if (ticker) (clearInterval(ticker), (ticker = null));
    const taken = await tracker.stopRecording();
    if (!taken) return;
    if (taken.frames.length < 2) return onattempt(null, "Inspelningen är tom.");
    scoring = true;
    try {
      const attempt = await scoreAttempt(
        taken.frames,
        taken.audio,
        taken.audioStart,
        sentence,
        hand() ?? "right", // read when the recording is sent, since it can be changed under Tecken
        camera.video!,
        lexicon.fps,
        unheardIsMiss,
      );
      onattempt(attempt, "");
    } catch {
      onattempt(null, "Bedömningen misslyckades. Spela in igen.");
    } finally {
      scoring = false;
    }
  }

  // The attempt ends by itself, so space only ever starts it over. preventDefault also stops it
  // clicking whichever button has focus.
  function onkeydown(event: KeyboardEvent) {
    if (event.code !== "Space" || (event.target as HTMLInputElement).type === "search") return;
    event.preventDefault();
    if (!event.repeat && tracker && !scoring) arm();
  }

  // The picture is outlined while the voice is being recorded and not before (user, 2026-09-24):
  // waiting for the learner to speak is not recording anything they would want back, and an outline
  // that is on throughout says nothing. Sök outlines it the same way while its own recording runs.
  $effect(() => void (camera.recording = phase === "speaking"));
  $effect(() => void (camera.listening = phase !== "idle"));
</script>

<svelte:window {onkeydown} />

<!-- Nothing is pressed to record and nothing is written about it (user, 2026-09-24): the recorder
     arms itself when a card appears, starts on the voice, ends on the silence after it and arms
     again when a wait runs long, and the pages arm it again after a verdict. What it is doing is the
     outline on the picture, so the phases are not written out beside it. Space starts a recording
     over, which is all the button that used to stand here did. -->
{#if silent}<p class="dim">{NO_MICROPHONE}</p>{/if}
