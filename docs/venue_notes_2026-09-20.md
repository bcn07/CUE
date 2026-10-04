# Venue notes, HackMIT 2026 (20 September 2026, 11:05)

A cleaned copy of the state-of-play note written for the team during the event.
Everything below was observed on the director Mac that morning. Local addresses,
the tunnel URL and the join code from the original note are removed; they have
all expired.

At the time the director was running the same code as commit 8c6f434 (the
first commit on this branch), and the test suite passed 199 tests.

## Why switching stopped working during rehearsal (live state at 11:05)

1. **Every camera's role was "audience".** `data/cameras.json` had A, B and D
   all as `audience`, and camera C had been removed. With no host, guest or wide
   camera, a name cue had nowhere to go and "back to me" had no host camera.
   Fix on `/setup`: one wide, one host, the rest guests, then Save.
2. **Only one camera was healthy, and it showed no faces.** Identity can only
   cut to a camera that currently shows the named person's face.
3. **Nobody was marked as host.** Tick "This person is the HOST" on `/setup`
   for whoever holds the mic.
4. **Faces need to be big enough.** The identity gate was lowered that morning
   from 6% to 4% of the frame width (28 px floor). At 6% it threw away every
   face on a phone held mid-room, even though those faces matched at 0.57 and
   0.63 against a 0.36 threshold. Keep phones within about two metres of the
   people they cover.

Fixed in code that morning: the desk now follows a tile click (a manual take did
not bump the decision counter the desk watches, so the desk kept showing the old
camera and swallowed later clicks). The director had also been left in HOLD,
because clicking a tile on the upstream desk means "take over".

## Built on the morning of 20 September

- **Sheet sign-ups enrol themselves** (`server/signup_sync.py`): the director
  polls the desk pages' Apps Script and turns each consenting row into a person
  with both photos and the camera the dashboard assigned. It never restarts the
  Deepgram stream. 25 tests.
- **Deepgram Flux** (`CUE_DEEPGRAM_MODEL=flux-general-en`): turn-based model on
  the v2 endpoint, one clause per sentence per turn. Measured on the real API:
  the final clause arrived 0.85 to 1.0 s after the audio ended (nova-3: 0.55 s,
  but nova-3 splits a sentence at a pause). Key terms update in place, so a
  roster change does not reconnect.
- **Mic selected by name** (`CUE_MIC_DEVICE`): a USB audio device had taken the
  default slot and the director was transcribing silence.
- **Desk: a tile for every camera.** Per role, the live camera takes the tile;
  extra cameras get their own tile (Guest 2, key 4).
- **"Let's look at the TV / screen / whiteboard / audience"** is handled by the
  rules, not only the LLM: it cuts to the camera whose role is `demo` (or
  `audience`), otherwise to the wide shot, never to a guest.
- **Scene tags** (`server/scene.py`, off by default, `CUE_SCENE=1`). Off because
  it keeps a second 3B model resident, and Ollama died once that morning with
  three models loaded, which also took the cue interpreter down.
- **LAN HTTPS server for the sign-up page** (`tools/serve_https.py`, from the
  upstream desk UI repo).

## Known limits at the time

- During the 10:17 to 11:00 session the interpreter's LLM calls averaged about
  5 s and timed out (Ollama had died and been restarted, with the vision model
  loaded alongside). The rules fast path still handled "please welcome <name>"
  without the LLM. `services.llm.errors` in `/api/state` shows this.
- Sheet sign-up had been verified only against a fake Apps Script and one real
  poll of the empty sheet.
- Venue Wi-Fi that isolates clients breaks the LAN sign-up link, and a quick
  tunnel's address changes whenever it restarts.
- Two phones on the same camera id put the second one into standby.

## Pre-show checklist

1. `/setup`: set camera roles (one wide, one host, guests), tick the host, Save.
2. Point the phones at people, close up. Check that `/setup` shows each camera
   as live and that the desk shows faces in the tiles.
3. Desk: the middle button should read "Take over" (CUE is directing), not
   "Let CUE direct" (manual).
4. Say "please welcome <enrolled name>" and watch the desk's Feed tab: a cue
   line, then a decision line naming the camera.
