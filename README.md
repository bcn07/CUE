# CUE

CUE is an AI live director: it listens to the host, works out who the show is
about at that moment, and cuts between cameras on its own. When it is not sure,
it holds the shot.

**HackMIT 2026 (MIT, 19–20 September 2026): 2nd place in the Deepgram track**
(Deepgram challenge "Build Something Worth Talking To"). The judges saw commit
`fa1f474`; the commits after it add documentation, credits and repository
clean-up only.

## What it does

Phones (or laptops) stream video to one Mac. The Mac transcribes the host's
microphone with Deepgram and turns each finished clause into a *meaning*: who is
being talked about, what the host intends, and whether it is happening now. It
recognises who is on which camera from enrolment photos. A director written in
plain code then picks the shot:

- **SHOW** the person being introduced or addressed
- **WIDE** for groups, applause, or when the right camera is not available
- **HOST** when the host takes the floor back
- **HOLD** when the meaning is unclear, the person is not on camera yet, or
  cutting would be too soon after the last cut

The programme output, the host's audio and every cut (with its reason) are
recorded. An operator can watch the reasoning live, take over, or run CUE in
ASSIST mode, where every cut becomes a suggestion to accept with one click.

## How a cut happens

Take "Please welcome Sarah Tan to the stage."

1. **Speech.** Deepgram (Flux or nova-3) streams the host's mic and finalises
   the clause. Roster names are sent as key terms so they are spelled right.
2. **Meaning.** A rule pass answers in about 0.1 ms when the pattern is
   unambiguous ("welcome" + a name as its object) and fires the cut at once. A
   local LLM (`qwen2.5:3b` on Ollama, JSON-schema output, about 1 s) follows and
   can correct it, and it catches phrasings the rules do not know. A
   deterministic guard turns anything past, future, negated, hypothetical, a
   question about someone, or an injected instruction into HOLD, so "Sarah will
   join us later" never cuts.
3. **Identity.** Every camera's latest frame is checked about 7 times a second
   with OpenCV's YuNet face detector and SFace embeddings (cosine ≥ 0.363, a
   0.06 margin over the runner-up, two consistent hits). An unknown or ambiguous
   face stays unknown.
4. **Decision.** `server/director.py` cuts to a healthy camera that has a fresh
   (≤ 4 s) confirmed match for Sarah. Without one it falls back to a fixed
   person-to-camera mapping, then the wide shot. A minimum shot length (2.5 s),
   a cue lifetime (3 s) and a rapid-cut guard stop it from flapping. A
   correction ("Sarah, actually Daniel") re-cuts once.
5. **Context.** A world state per camera (who is visible, mouth motion, shot
   type, frozen frames) and a planner with readable reason codes handle the
   cases a name alone cannot: a question without a name goes to the addressee,
   "a round of applause" goes to the audience camera, "here's how it works"
   goes to the demo camera, and "mm-hmm" never cuts.

## Architecture

```
Phones (Safari, /cam) ──WebRTC──► LiveKit Cloud room ──┐
Laptops (camera/stream_camera.py) ──JPEG over WS──────┤
                                                       ▼
                      Director Mac: FastAPI app (server/app.py)

  host mic ──► Deepgram STT ──► clauses ──► rules fast path + local LLM (Ollama) ──► cue
  frames ────► YuNet + SFace ──► who is on which camera ───────────┐
         └───► observers (mouth motion, shot type, frozen frames) ─┴─► world state

  cue + world state ──► planner (scores, reason codes) ──► director (pure code)
                                                           SHOW / WIDE / HOST / HOLD
                                                                       │
              operator desk (/desk) · setup (/setup) · recorder (program.mp4, audio.wav, cuts.jsonl)
```

Phone video goes through LiveKit; the director joins the room subscribe-only
and re-encodes frames into the same camera path the laptops use. The master
audio is always the Mac's microphone, never a camera.

## Measured results

All measured on one M2 Pro MacBook (16 GB) on 19–20 September 2026.

| What | Result | Source |
|---|---|---|
| Interpreter, held-out set (40 lines, run once) | `qwen2.5:3b`: 38/40 (95%), **0 wrong cuts**, 1.0 s p50 / 1.1 s p95 | [`reports/llm_eval_heldout_qwen2.5-3b_20260919_202004.json`](reports/llm_eval_heldout_qwen2.5-3b_20260919_202004.json) |
| Interpreter, dev set (58 scorable lines) | `qwen2.5:3b`: 54/58 (93%), 2 wrong cuts | [`reports/llm_eval_adversarial_qwen2.5-3b_20260919_201924.json`](reports/llm_eval_adversarial_qwen2.5-3b_20260919_201924.json) |
| End of speech to cut, live mic | 1.17 s with nova-3 (one introduction played by text-to-speech into the Mac's mic; the LLM confirmed 1.06 s later) | [operator guide, "Evidence so far"](docs/OPERATOR_GUIDE.md#evidence-so-far-this-macbook-no-keys) |
| Deepgram Flux finalisation | final clause 0.85–1.0 s after the audio ended (nova-3: 0.55 s, but nova-3 can split a sentence at a pause) | [operator guide](docs/OPERATOR_GUIDE.md#evidence-so-far-this-macbook-no-keys) |
| 3-minute soak | 30/30 lines correct, 0 wrong cuts, 20/20 expected cuts, failover 0.32 s after the live camera died | [`reports/soak_2026-09-19_3min.md`](reports/soak_2026-09-19_3min.md) |
| Face identity | 11.8 ms per frame on average over 3,545 frames (decode + detect + embed) | [`reports/soak_2026-09-19_3min.md`](reports/soak_2026-09-19_3min.md) |
| Tests | 201 unit and end-to-end tests | [`tests/`](tests/) |

The other models tried (`llama3.2:3b`, `qwen2.5:7b`) are in the same
`reports/` folder. The 7B model matched the 3B on the held-out set but took
2.3 s p50, too slow for a 3 s cue lifetime. The soak used typed lines, the
rules interpreter only and synthetic cameras streaming the test photos; its cut
log and recording summary are in `reports/` too.

## Running it

Developed and tested on macOS (Apple Silicon) with Python 3.12. You need
Python 3.10+, [Ollama](https://ollama.com) for the local interpreter, a
[Deepgram](https://deepgram.com) API key for live speech, and optionally a
[LiveKit Cloud](https://livekit.io) project for phone cameras and ffmpeg for
the recording's audio mux.

```bash
brew install ollama ffmpeg          # ffmpeg is optional
git clone https://github.com/bcn07/CUE.git
cd CUE
cp .env.example .env                # add DEEPGRAM_API_KEY; LIVEKIT_URL/KEY/SECRET for phones
./run_server.sh                     # creates .venv, installs deps, downloads the face models,
                                    # starts Ollama and pulls qwen2.5:3b, serves on :8000
```

Then:

1. Open `http://localhost:8000/setup`. Add people (name, aliases the host might
   say, 2–5 clear photos), tick the host, and give each camera a role: `wide`,
   `host`, `guest`, `audience` or `demo`.
2. Connect cameras. A laptop on the same network runs the line `/setup` shows,
   e.g. `./camera/run_camera.sh --server ws://<director-ip>:8000 --cam B --code <join-code>`.
   A phone needs HTTPS: run a tunnel to port 8000 (cloudflared or ngrok), paste
   its address on `/setup`, and open the per-camera link in Safari. Through the
   tunnel only the camera page, its socket and the join-code-gated LiveKit token
   route are reachable.
3. Direct from `http://localhost:8000/desk` (operator desk) or
   `http://localhost:8000/` (director view with the reasoning panels). AUTO
   cuts, ASSIST suggests, HOLD blocks automatic cuts; REC records to
   `data/recordings/`.

Without cameras, a mic or keys:

```bash
.venv/bin/python -m pytest -q                                                    # 201 tests
./run_server.sh                                                                  # terminal 1
.venv/bin/python tools/fake_camera.py --cam A --pattern                          # terminal 2
.venv/bin/python tools/fake_camera.py --cam B --image tests/fixtures/obama2.jpg  # terminal 3
# on /setup, enrol "Barack Obama" with tests/fixtures/obama.jpg, then:
.venv/bin/python tools/say.py "Please welcome Barack Obama!"
```

The [operator guide](docs/OPERATOR_GUIDE.md) covers everything else: every
setup step, iPhone setup, virtual pan and zoom, the desk, guest sign-up,
recording, the bridge to the team's GUI, all tunables, and the full evidence
log. [`docs/CUE_CONTEXT_AWARE_DIRECTOR_SPEC.md`](docs/CUE_CONTEXT_AWARE_DIRECTOR_SPEC.md)
is the design spec for the planner.

## Limitations

- **Built in a weekend, tested on one machine.** All numbers come from one
  MacBook and short sessions. The eval sets are small (40 held-out and 58 dev
  lines), English only, and written by the team.
- **The live latency figures are single sessions** with text-to-speech audio
  (played into the Mac's microphone for nova-3, streamed straight to the API for
  Flux), not a real host in a noisy hall. The soak used synthetic cameras and
  typed lines.
- **The local LLM is the bottleneck under load.** At the venue, with Ollama
  restarted and a vision model loaded alongside, LLM calls averaged about 5 s
  and timed out during one 43-minute session; the rule fast path kept cutting
  ([venue notes](docs/venue_notes_2026-09-20.md)). The optional scene tagger is
  off by default for this reason.
- **Faces must be reasonably large.** Identity ignores faces narrower than 4%
  of the frame (28 px minimum), so phones need to be within about two metres of
  the people they cover. An unconfirmed face means a wide shot, not a guess.
- **AUTO is not formally validated.** The AUTO gates in the spec (§10) were
  not exercised on a real setup; the operator guide recommends ASSIST until
  they are.
- **Not verified live:** the OpenAI interpreter path and the OpenAI vision
  fallback; guest sign-up with real sign-ups (only a fake Apps Script and one
  poll of an empty sheet); and phone video over LiveKit has no saved
  measurements (it was verified with a synthetic publisher and headless Chrome).
- **Trusted network only.** The director, setup page and API have no login.
  Cameras need a join code, and a public tunnel exposes only the camera page,
  but anyone on the LAN can reach the control surface.
- `tools/eval_llm.py` reads the evaluation corpora from the team repo
  (`scripts/data/` in [chocoHacks33/CUE](https://github.com/chocoHacks33/CUE)),
  cloned next to this one as `../hackmit_2026_cue`.

## Team and credits

CUE was built at HackMIT 2026 by:

- Brian Nwaghodoh ([@bcn07](https://github.com/bcn07)), who wrote the director in this repository
- Prakash ([@chocoHacks33](https://github.com/chocoHacks33))
- Aditi Muduganti ([@mudadit26](https://github.com/mudadit26))
- Hema Dassani ([@hemadassani](https://github.com/hemadassani))

The team's main repository, with the React production GUI and control-plane API
that `server/bridge.py` talks to, is
[chocoHacks33/CUE](https://github.com/chocoHacks33/CUE).

The operator desk and guest pages come from
[hemadassani/cue-desk-ui](https://github.com/hemadassani/cue-desk-ui):

- `server/static/desk.html`: the desk UI by **Hema Dassani**, adapted here to
  receive camera frames over `/ws` and to show a tile for every camera.
- `server/static/dashboard.html`, `server/static/signup.html`,
  `server/static/config.example.js` and `tools/serve_https.py`: by **Aditi
  Muduganti**, in the same repository.

The test photos in `tests/fixtures/` are official White House photographs by
Pete Souza, taken from the examples of
[ageitgey/face_recognition](https://github.com/ageitgey/face_recognition); see
[`tests/fixtures/README.md`](tests/fixtures/README.md).

## Built with

- [Deepgram](https://deepgram.com): streaming speech-to-text (Flux on the v2
  endpoint, or nova-3), with roster key terms
- [LiveKit](https://livekit.io): WebRTC transport from the phones (LiveKit
  Cloud, `livekit` and `livekit-api` for Python, `livekit-client` in the browser)
- [Ollama](https://ollama.com) with [Qwen2.5 3B](https://ollama.com/library/qwen2.5):
  the local interpreter, through the OpenAI-compatible API with a JSON schema
- OpenCV [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface):
  face detection and recognition
- [FastAPI](https://fastapi.tiangolo.com) and Uvicorn: the director server,
  WebSockets and the operator pages
- ffmpeg: the live fragmented-MP4 recording and the audio mux

## Repository layout

```
server/        director server: speech, interpreter, identity, world state, planner, director, recorder, LiveKit, desk adapter
server/static/ director view, setup page, camera page, and the desk / dashboard / sign-up pages
camera/        laptop camera script with macOS/Linux and Windows launchers
tools/         fake cameras, typed input, soak test, key check, interpreter eval, HTTPS dev server
tests/         unit and end-to-end tests; face fixtures
reports/       soak report, cut log, recording summary and interpreter evals
docs/          operator guide, director spec, venue notes, iPhone camera plan
```
