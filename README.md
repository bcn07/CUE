# CUE one-shot: context-aware live director (standalone experiment)

Three laptops stream their webcams to one MacBook. The MacBook listens to the
host's mic, turns every finished clause into a *meaning* (who / intent / timing),
recognises who is on which camera from the photos you uploaded, and a
deterministic director cuts the broadcast: **SHOW** the person being addressed,
**WIDE** for groups, **HOST** when the host takes back the floor, **HOLD** when
unsure. Deciding *not* to cut is a feature. The program output, the master audio
and every cut are recorded.

This folder is self-contained. It does not touch the team repo and is not pushed anywhere.

```
laptop A  ─┐  JPEG frames over WebSocket (LAN)
laptop B  ─┼────────────►  MacBook  server/app.py  ──►  http://localhost:8000  (program + judge panel)
laptop C  ─┘                 │  mic ──► Deepgram ──► clauses ──► rules fast path + LLM meaning
                             │  frames ──► YuNet + SFace (local, ~12 ms/frame) ──► "camera B shows Sarah"
                             │  director.py (pure code) ──► SHOW / WIDE / HOST / HOLD
                             └─ recorder.py ──► data/recordings/<ts>/program.mp4 + cuts.jsonl
```

## 0. What you need

| Where | Needs |
|---|---|
| MacBook (director) | Python 3.10+ (uv or python3), this folder, Wi-Fi shared with the laptops, `OPENAI_API_KEY` + `DEEPGRAM_API_KEY` in `.env`, ffmpeg (optional, for muxing audio into the recording) |
| Each camera laptop | Python 3.8+ with `pip`, a webcam, same Wi-Fi. The launcher installs its own 2 packages. |

Keys are read from `cue_oneshot/.env` (created from `.env.example` on first run); a name that is
still empty is also looked up in the team repo's `../hackmit_2026_cue/.env`, and the startup log says
which file each key came from (never the value). If a key is missing the system still runs in a
degraded mode and says so on screen: no OpenAI key → the
deterministic rule interpreter only (no LLM, no VLM); no Deepgram key → no live transcription, use
the typed test input (labelled "typed", never shown as live speech).

## 1. MacBook: start the director

```bash
cd cue_oneshot
cp .env.example .env      # first time only: paste OPENAI_API_KEY and DEEPGRAM_API_KEY
./run_server.sh
```

First run creates `.venv`, installs deps, downloads the two face models (~39 MB). It prints:

```
  CUE director  : http://localhost:8000/
  Setup page    : http://localhost:8000/setup
  Camera laptops: ./camera/run_camera.sh --server ws://<your-lan-ip>:8000 --cam B
```

macOS asks for **Microphone** permission for your terminal the first time the mic starts
(Deepgram or REC), and **Camera** permission if you stream from the Mac itself. If the port is
already taken the launcher says which process owns it and stops instead of silently serving stale code.

Check the keys in 10 seconds (never prints them): `.venv/bin/python tools/smoke_keys.py`

## 2. Setup page (before the show): scripts, faces, cameras

Open `http://localhost:8000/setup`.

1. **People**: name, aliases the host might say ("Sarah", "Ms Tan"), role, tick *HOST* for the
   person on the mic, upload 2–5 clear photos each. The page reports how many photos produced a
   usable face. A person with 0 faces can only be found through the fixed camera mapping.
2. **Event script**: paste the run of show (or upload .md/.txt). The interpreter reads it as
   context (segment order, who speaks when); the live words always win.
3. **Cameras**: for each camera id set a label and a role: `wide` (the safe shot), `host`
   (where the host sits), `guest`. Optionally a *fixed person* used only when the live face match
   cannot confirm that person. Labels saved here win over what a laptop announces.

Everything is saved under `data/` and survives restarts.

## 3. Camera laptops: one command each

Copy the exact line shown on the setup page. Generic form (replace the IP with the MacBook's):

macOS / Linux
```bash
# copy the cue_oneshot/camera directory to the laptop (AirDrop / USB / git), then:
./camera/run_camera.sh --server ws://10.189.100.223:8000 --cam A --code 482913
./camera/run_camera.sh --server ws://10.189.100.223:8000 --cam B --code 482913
./camera/run_camera.sh --server ws://10.189.100.223:8000 --cam C --code 482913
```

Windows
```bat
camera\run_camera.bat --server ws://10.189.100.223:8000 --cam C --code 482913
```

The join code is printed on the setup page next to these commands.

Only the `camera/` directory is needed on a laptop (`stream_camera.py`, `requirements-camera.txt`,
`run_camera.sh` / `run_camera.bat`). First run creates a venv and installs `opencv-python` and
`websockets`; an interrupted install is retried next time. Useful flags: `--list-devices`,
`--device 1` (external webcam), `--width 854 --height 480`, `--fps 20`, `--quality 80`, `--preview`.
Frames are letterboxed, never stretched. Defaults are 640x360 @ 15 fps ≈ 2.5 Mbit/s per camera
(measured 2.4 Mbit/s from a 1080p FaceTime camera), which survives hackathon Wi-Fi. A stalled link is
detected within 5 s and the script reconnects. The camera id (`--cam`) is created on the director
automatically; set its role on the setup page. One id per laptop: a second connection with the same
id replaces the first (the older stream is closed, never interleaved).

No Python on a laptop? Open `http://<mac-ip>:8000/cam?cam=B` in Chrome. Chrome blocks webcams on plain
http for non-localhost pages, so once per laptop enable
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` for `http://<mac-ip>:8000` (the page explains
this). Keep that tab visible: Chrome throttles hidden tabs.

## 3b. iPhones as cameras (HTTPS tunnel, join code, virtual pan/zoom)

Phones need HTTPS for the camera, so the director gets a public HTTPS address through a free
Cloudflare quick tunnel (no account). On the Mac:

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8000      # prints https://<words>.trycloudflare.com; keep it running
```

Paste that address into the box at the top of `/setup`; it prints one link per camera, e.g.
`https://<words>.trycloudflare.com/cam?cam=B&code=482913`. Open the link in Safari on the phone, tap
Start, allow the camera. Rear camera by default, a lens picker (wide / ultra wide / telephoto where
iOS exposes them), 1080p capture, screen kept awake, LIVE / STANDBY / OFFLINE shown on the phone,
automatic reconnect. Phones can be on mobile data; the tunnel also sidesteps venue Wi-Fi that blocks
device-to-device traffic (laptops can use the same link in Chrome).

Security, on by default:

- **Join code**: `/ingest` requires `code=<6 digits>`; the code is shown on `/setup`, is per event
  and can be rotated there (laptop command: `--code 482913`). A wrong code is refused and logged.
- **Tunnel-only surface**: for a `*.trycloudflare.com` hostname (or `CUE_PUBLIC_URL`), only `/cam`
  and the camera socket exist; the director, the setup page, every API and the director socket
  return 404 through the tunnel. They stay on the LAN.
- **One publisher per camera id**: a second device with the right code waits in STANDBY and only
  replaces the live one after you press REPLACE on that camera's tile (or when the live one drops).
- Video only from phones; the master microphone stays the Mac's. Close the tunnel after the show.

Virtual PTZ: the phone captures full resolution and sends a cropped window (default 960x540). On the
director's multiview, drag a tile to pan, scroll to zoom (1x to 3x, clamped inside the frame), `⟲`
resets, `1 2 3` recall presets (`save` then a number stores the current framing). The phone eases to
the new framing over 400 ms. Camera moves are per camera and survive reconnects.

Phone setup: plugged in, Low Power Mode off, Auto-Lock Never, Do Not Disturb on, Guided Access to
pin Safari, landscape on a stand, Safari in the foreground (iOS pauses the camera in the background;
the page shows OFFLINE and reconnects when it returns).

## 4. Show time: `http://localhost:8000/`

- **Program** (big): the camera that is live, with the reason for the last decision.
- **Multiview**: every camera with health, fps, and the faces it recognises (green boxes + name +
  similarity). Click a camera to take it manually.
- **Transcript**: interim text in grey, finished clauses in white. Typed test lines are labelled.
- **Meaning**: subject, intent, timing, action, the exact words that decided it, source (`rules-fast`,
  `llm`, `rules`), interpreter latency.
- **Decision**: TAKE / STAY / SLATE, camera, evidence (`identity`, `static`, `wide`, `host-role`,
  `manual`) and a human-readable reason.
- **Latency**: interpret ms, LLM ms, clause→cut ms, speech→cut ms (estimated from Deepgram audio
  time), identity ms, decide ms, with p50/p95.
- **Controls**: AUTO / HOLD, WIDE, HOST, TAKE per camera, **REC**. Keyboard: `H` toggles hold, `W`
  wide, `A`/`B`/`C` take that camera (plain key presses only; Cmd/Ctrl combos are ignored).
- **REC** records `data/recordings/<timestamp>/`: `video.mp4` (H.264 in a fragmented MP4 written
  live through ffmpeg, so it stays playable even if the server dies mid-show), `audio.wav` (master
  mic, continuous, never touched by a video cut, padded with leading silence if the mic opens late),
  `program.mp4` (video + audio muxed at stop; the intermediates are kept next to it), `cuts.jsonl`
  (every program change and slate with offset, camera, evidence, reason, cue, latency),
  `transcript.jsonl`, `summary.json`. Without ffmpeg the video is written by OpenCV and finalised only
  on stop. `CUE_AUTO_RECORD=1` starts recording at boot. Recordings are listed on the setup page and
  served under `/recordings/`; the REC pill warns when the mic has been silent.

## How a cut happens (and why it often does not)

1. Deepgram finalises a clause (~300 ms after the host pauses). Every finished clause is acted on;
   clauses of one utterance share an id so a correction ("Sarah, actually Daniel") gets **one** fast re-cut.
2. Meaning: a deterministic rule pass answers in ~0.1 ms when the pattern is unambiguous. A cut needs
   the name to be the *object* of a stage verb ("please welcome our next guest, Sarah") or a direct
   address with a handoff tail ("Daniel, what do you think?"). It fires the cut immediately
   (`rules-fast`); the LLM (`gpt-4.1-mini` structured output, ~0.5–1 s, run off the transcription path)
   follows and can correct it. Future / past / negated / hypothetical / retractions ("stay where you
   are") / questions about a person / injected instructions never produce NOW, so they never cut.
   On the team's corpora the rules never cut wrongly (adversarial: 0 wrong cuts, 40/46 exact, 10/14
   introductions caught; held-out: 0 wrong cuts, 30/40 exact, 7/14 caught). Recall is the LLM's job.
   The held-out set was read for scoring only; one wrong cut it exposed ("Over to X, actually let's
   start with Y") was fixed as a general rule (a correction marker ends a name chain), not by adding
   the sentence.
3. Identity: each camera's latest frame is checked ~7×/s locally (YuNet + SFace, cosine ≥ 0.363 and
   a 0.06 margin over the runner-up, 2 consistent hits). An unknown or ambiguous face stays unknown.
   Optional VLM fallback (`CUE_VLM=1`, needs the OpenAI key) asks the vision model only when the
   local match cannot confirm anyone, at most every 4 s per camera, never on the cut path.
4. Director (pure code, `server/director.py`): a named SHOW needs a healthy camera with a *fresh*
   confirmed identity of that person (≤ 4 s), preferring the camera already live; else the fixed
   mapping, unless that camera currently shows somebody else; else WIDE; else SLATE. Minimum shot
   2.5 s, bypassed once per utterance for a real correction (a different person) and for failover
   when the live camera dies. A cue older than 3 s is dropped. A cue that could not be executed yet
   stays pending for its lifetime, is re-evaluated when identity changes or the shot may end, and is
   dropped when the host retracts, defers, or starts a new utterance, or when the LLM disagrees with
   the fast path.
5. Manual HOLD beats everything automatic; failover to a healthy camera still works.

Tunables live in `.env` (`CUE_MIN_SHOT_S`, `CUE_IDENTITY_MAX_AGE_S`, `CUE_CUE_LIFETIME_S`,
`CUE_LLM_MODEL`, `CUE_VLM`, `CUE_FAST_PATH`, `CUE_MIC_DEVICE`, `CUE_RECORD_*`, ...).

## Open-source LLM instead of OpenAI (what this build uses by default)

No OpenAI key is needed. `.env` points the interpreter at a local Ollama server
(`CUE_LLM_BASE_URL=http://127.0.0.1:11434/v1`, `CUE_LLM_MODEL=qwen2.5:3b`); `./run_server.sh` starts
Ollama and pulls the model if needed (`brew install ollama` once). The same OpenAI SDK talks to it
through the chat-completions API with a JSON schema; the model is warmed at boot and kept resident.
On top of the model output a deterministic guard applies: a deferral / negation / past / question /
retraction / injection found by the rule engine turns NOW into HOLD; a target the clause never
names (by alias or role phrase) is dropped; HOST needs a real return-to-host phrase; a correction
the model read as a group is resolved by the rules. Measured with `tools/eval_llm.py` on the team's
corpora (the held-out set was run once, for the release number; guard rules were designed on the
dev set and on general small-model failure modes):

| model (M2 Pro, 16 GB) | dev set (58 scorable) | held-out (40) | wrong cuts dev / held-out | latency p50 / p95 |
|---|---|---|---|---|
| qwen2.5:3b (default) | 93% | 95% | 2 / 0 | 1.0 s / 1.1 s |
| llama3.2:3b | 97% | 92% | 1 / 1 | 1.0 s / 1.3 s |
| qwen2.5:7b | 97% | 95% | 1 / 0 | 2.3 s / 2.7 s (too slow for a 3 s cue lifetime on this Mac) |
| rules alone (fast path) | 40/46 exact, 0 wrong cuts | 30/40 exact, 0 wrong cuts | 0 / 0 | 0.1 ms |

The M1 gate ("90%+ on adversarial with p95 latency measured") passes for both models. The 4
dev cases the corpus marks DUPLICATE_NAME need a two-Sarah roster and are excluded. Live check on
this Mac with fake cameras: "Big hand for Barack Obama, everybody!" (a phrase the rules do not
know) cut to Barack's camera through the model; a past mention held; "Thank you Barack" returned to
the host. To use OpenAI instead, remove the two `CUE_LLM_*` lines from `.env` and set
`OPENAI_API_KEY`; the vision fallback (VLM) only runs with OpenAI.

## Driving the team's GUI (bridge to the team control plane)

The one-shot can be the brain behind the team's React GUI instead of its own program view. Every
program change the director makes is mirrored to the team backend (`apps/api` in the CUE repo) as a
producer take on `POST /api/v1/events/{event}/take`; one-shot HOLD/AUTO become team `MANUAL_HOLD` /
`ASSIST`. The team GUI receives the `render.command` on its control websocket and cuts its LiveKit
video. Nothing in the team repo changes; the bridge only speaks the team's public API.

```bash
# terminal 1: team API (its own venv), default port 8000
cd ~/Developer/hackmit_2026_cue/apps/api && .venv/bin/uvicorn cue_api.main:app --host 0.0.0.0 --port 8000
# terminal 2: team web GUI
cd ~/Developer/hackmit_2026_cue && npm run dev:web            # http://localhost:5173, pair the laptops as usual
# terminal 3: the one-shot brain on another port, pointed at the team event
cd ~/Developer/cue_oneshot
CUE_PORT=8001 CUE_TEAM_API=http://127.0.0.1:8000 CUE_TEAM_EVENT_ID=<the event id the GUI uses> ./run_server.sh
```

`CUE_TEAM_PRODUCER_SECRET` defaults to `CUE_PRODUCER_SECRET` from the team repo's `.env`;
`CUE_TEAM_CAMERA_MAP=A=CAM-WIDE,B=CAM-GUEST,C=CAM-HOST` maps one-shot camera ids to the team's. The
stream epoch for a take comes from the team's camera bindings, so the compositor accepts it.

How the team compositor treats these takes (its `controlAdapter.ts` classifies by reason code; ours
are `CUE_<evidence>`, POLICY origin, never `MANUAL`):

- **ASSIST** (default, `CUE_TEAM_RESUME_MODE=ASSIST`): the one-shot's decisions appear in the team
  GUI as suggestions and the operator presses TAKE. Safe: nothing cuts on air without a human.
- **AUTO** (`CUE_TEAM_RESUME_MODE=AUTO`): the decisions cut on air, subject to the compositor's 2.5 s
  minimum shot. In AUTO the team's own CLane policy can also cut, so the two brains would fight unless
  C's lane is off. The team's plan also has a one-production-pipeline rule: running the one-shot's
  Deepgram + LLM next to C's lane is a second pipeline. Both are team decisions, not defaults.

One-shot HOLD maps to the team's `MANUAL_HOLD`; one-shot AUTO maps to the resume mode above. The
director view shows a "Team GUI" pill with the team mode, live camera and takes sent. Face identity
on the one-shot side needs the laptops to also run `camera/stream_camera.py` (LiveKit frames are not
visible to it); without that, use the fixed person-to-camera mapping on the setup page, the same
role-based fallback the team's demo uses.

Verified on this Mac against the real team API (`codex/stage-5-integration`): a one-shot take
became a `render.command` for `CAM-GUEST` on the GUI's control socket in 5 ms, HOLD and AUTO mapped
to `MANUAL_HOLD` and `ASSIST`, and a competing producer's stale revision (HTTP 409) was retried
correctly. Not verified: a browser showing the team GUI cutting LiveKit video from real laptops.

## Test it without laptops, mic or keys

```bash
.venv/bin/python -m pytest -q                       # 122 tests: assembler, rules, director, pending cues, identity, recorder, speech, team bridge, end-to-end
./run_server.sh                                     # terminal 1
.venv/bin/python tools/fake_camera.py --cam A --pattern                          # terminal 2
.venv/bin/python tools/fake_camera.py --cam B --image tests/fixtures/obama2.jpg  # terminal 3
.venv/bin/python tools/say.py "Please welcome Barack Obama!"                     # after enrolling Barack on /setup
CUE_PORT=8001 CUE_DATA_DIR=/tmp/cue-soak ./run_server.sh                         # a scratch director for the soak (terminal 4)
.venv/bin/python tools/soak.py --server http://127.0.0.1:8001 --minutes 3 --setup  # soak + failover test; --setup enrols two test people and rewrites cameras, so never point it at your real show
```

The end-to-end test starts the real server, streams real photos as cameras, enrols people through
the upload API and drives the pipeline with typed clauses. It checks: identity lands on the right
camera, SHOW cuts to it, a cue inside the minimum shot is held then executed, FUTURE/questions never
cut, a mid-utterance correction re-cuts once, HOLD blocks, operator labels survive reconnects, a stale
connection closing does not kill a reconnected camera, recording produces playable files and a cut
log, person ids cannot escape `data/people`, and a dead live camera fails over to wide.

`tools/soak.py` runs three fake cameras, one browser-like client and a 12-line host script in a loop,
kills the live guest camera once and brings it back, samples the server's memory, and writes a report.

## Evidence so far (this MacBook, no keys)

| what | result |
|---|---|
| unit + end-to-end tests | 122 passed (`.venv/bin/python -m pytest -q`): assembler, rule parser (incl. 40+ adversarial sentences from two review rounds), director, pending-cue semantics, identity, recorder (incl. mid-recording playability), Deepgram protocol against a fake server, end-to-end |
| live feed | this Mac's FaceTime camera via `camera/stream_camera.py`: 153 frames in 10 s at 15.0 fps, 2.4 Mbit/s, face detected, auto-took wide, SLATE on disconnect |
| face identity | same person 0.74 cosine, different people 0.10–0.23; YuNet 640x360 ≈ 5 ms; decode+detect+embed ≈ 12 ms/frame avg |
| 3-min soak (typed lines, rules only, recording on) | `reports/soak_2026-09-19_3min.md`, final build: 30/30 lines correct, 0 wrong cuts, 20/20 expected cuts reached, clause→program 1.2 ms p50 / 1.8 ms p95, identity 11.8 ms avg over 3545 frames, both people confirmed 0.21 s after the cameras appeared, failover after the live camera died 0.32 s, camera back + identity re-confirmed 0.32 s, cameras in 15/15/15 fps, UI relay out 15/15/15 fps, RSS 292 → 140 MB |
| recording during that soak | 185 s `program.mp4` (H.264 fragmented + AAC, playable in Chrome and while still being written), 2762 frames, 19 cuts in `cuts.jsonl` (`reports/soak_2026-09-19_cuts.jsonl`), 184 s of continuous audio aligned with 0.4 s of leading silence for the mic start, every transcript line labelled `typed` |
| live speech | Deepgram nova-3 from the MacBook mic: exact transcript of a spoken introduction, cut 1.17 s after end of speech, LLM confirmation 1.06 s later; room chatter -> HOLD only |
| review | two rounds of read-only review by independent reviewers (11 reviewer passes, 37 agents in round two with adversarial verification of each finding). Round one: path traversal, stale-connection ownership, label clobbering, hotkey modifiers, wrong-cut rule patterns, pending-cue retraction, Deepgram keepalive, LLM blocking the transcript loop. Round two: an older pending cue firing after a newer satisfied cue, the live wide camera absorbing every SHOW as "already live", thank-you anywhere overriding an introduction, corrections retargeting on grammar-free markers, deferrals like "after Daniel", a stalled browser freezing everyone's state, exclusive camera ownership, LLM errors mistaken for verdicts, clause ordering of late LLM verdicts, unplayable recordings after a crash, port/.env mismatch, partial model downloads. Every confirmed finding is fixed and covered by a test. |

**Live speech, verified on this Mac (2026-09-19 evening, Deepgram key present)**: the Mac's
text-to-speech spoke "Please welcome Barack Obama!" into the MacBook microphone; Deepgram (nova-3)
transcribed it exactly, the rule fast path fired SHOW, the director cut to Barack's camera 1.17 s after
the words ended, and the local LLM confirmed the meaning 1.06 s later. A past-tense mention held; a
thank-you returned to the host. In a noisy room, 40 s of unrelated conversation produced only HOLDs.
Roster names and aliases are sent to Deepgram as key terms (nova-3 `keyterm`), and the stream
reconnects in ~0.25 s when the roster changes so the terms stay current. Set `CUE_MIC_DEVICE` to the
index or name printed by `tools/smoke_keys.py` if the system default input is not the right mic.

**Not verified live**: the OpenAI path (no key; the local model is used instead) and the VLM fallback
(OpenAI only). Both fail safe: a failed interpretation is a HOLD.

## Layout

```
run_server.sh            MacBook launcher (venv, deps, models, LAN URLs)
server/app.py            FastAPI: /ingest (cameras) /ui (browser) /api/* (setup, control, typed input, record)
server/semantics.py      meaning: rule fast path + OpenAI structured output; validate()
server/director.py       deterministic SHOW/WIDE/HOST/HOLD, min shot, correction, failover
server/identity.py       YuNet + SFace enrolment, matching, per-camera presence tracker, worker thread
server/recorder.py       program video + continuous master audio + cuts.jsonl, ffmpeg mux
server/vlm.py            optional vision-model fallback (rate-limited)
server/bridge.py         mirrors cuts and HOLD/AUTO to the team backend's control plane (team GUI)
server/speech.py         mic (sounddevice) -> Deepgram WebSocket -> clause assembler
server/assembler.py      Deepgram messages -> clauses / utterance ids (pure)
server/static/           index.html (director), setup.html, cam.html (browser camera fallback)
camera/                  laptop script + bootstrap wrappers + its 2-line requirements
tools/                   fake_camera.py, say.py, soak.py, smoke_keys.py
tests/                   122 tests incl. end-to-end; fixtures are public-domain portraits
reports/                 soak report, cut log and recording summary from this MacBook (2026-09-19)
data/                    roster, photos, script, camera config, recordings/, reports/ (created at runtime)
models/                  face_detection_yunet_2023mar.onnx, face_recognition_sface_2021dec.onnx
```
