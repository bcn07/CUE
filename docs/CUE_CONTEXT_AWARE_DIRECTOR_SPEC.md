# CUE — Fully Context-Aware AI Director

## Implementation specification for Codex

This document defines the intended intelligence of CUE. The product is not a
name-triggered camera switcher and must not be implemented as a collection of
rigid rules such as “if Saerah is mentioned, show Saerah.”

CUE is an AI live director. It continuously observes every available camera,
listens to the programme conversation, remembers what has happened, anticipates
what is likely to happen next, and selects the shot that best communicates the
story to the audience.

The target setup has three Windows camera laptops and one central Mac:

| Source | Intended role |
|---|---|
| `CAM-HOST` | Host/presenter view and the only master microphone |
| `CAM-GUEST` | Guest, interviewee or featured-person view |
| `CAM-WIDE` | Establishing, group, movement, demonstration and safety view |
| Mac | Receives every feed, maintains context and renders the programme output |

The master audio must remain continuous when the video changes. Camera
selection must never restart, replace or follow a different microphone.

---

## 1. Product behaviour

At every decision point, CUE should answer:

> Given everything visible, audible and previously established, what should the
> audience see now—and why?

The answer may be:

- remain on the current shot;
- cut to the active speaker;
- anticipate the next speaker;
- show the person reacting rather than the person speaking;
- show an object or demonstration;
- use the wide shot to explain movement or relationships;
- stay on an emotionally important moment despite competing activity;
- choose a healthy substitute when the ideal camera is unavailable;
- hold the current shot because cutting would be distracting;
- recommend a shot for operator approval when confidence is insufficient.

The best decision is frequently **not** “show whoever is speaking” and almost
never “show whoever was named most recently.”

---

## 2. Non-goals and prohibited shortcuts

Do not reduce the director to any of these behaviours:

```text
if transcript contains person_name: cut(person_camera)
if camera has loudest audio: cut(camera)
if active speaker changes: cut immediately
rotate cameras every N seconds
always show the host for questions and guest for answers
ask an LLM for a camera name with no persistent state
```

Names, audio activity, script lines and timing are evidence—not commands. A
person may be mentioned while somebody else owns the important visual moment.
A speaker may make a short acknowledgement that does not deserve a cut. A
silent reaction may be more valuable than the current speaker.

The script is an optional prior. The live event is authoritative. If the event
deviates from the script, CUE must follow reality.

---

## 3. Continuous context model

CUE must maintain one evolving world state rather than make isolated decisions
from individual transcript fragments.

### 3.1 Per-camera state

For every camera, continually estimate:

- which people are visible and where they are in frame;
- identity and identity confidence, when consented identity is enabled;
- whether each person is speaking, listening or preparing to speak;
- face visibility, body orientation, gaze direction and attention;
- meaningful gestures such as pointing, waving, raising a hand or reaching;
- facial reaction and rough emotional salience;
- whether an object, screen or demonstration is visible;
- movement such as entering, leaving, standing, sitting or crossing the stage;
- shot composition: close-up, medium, group, wide, object/detail or unusable;
- focus, exposure, obstruction, camera shake and frozen-frame status;
- frame freshness, stream health, latency and confidence.

Visual understanding should be temporal. “Saerah is reaching toward the
prototype” is more useful than three unrelated frame captions describing the
position of her hand.

### 3.2 Conversation state

Continuously estimate:

- current speaker and confidence;
- who is being addressed, even when no name is spoken;
- conversation topic and current subtopic;
- dialogue act: introduction, question, answer, rebuttal, joke, explanation,
  demonstration, transition, conclusion or audience interaction;
- expected next speaker;
- whether speech is overlapping, interrupted or merely back-channel feedback;
- references such as “she,” “his earlier point,” “both of you” or “the person
  who designed this”;
- emotional and narrative importance;
- whether a sentence is complete or the speaker is likely to continue;
- planned segment and actual live segment;
- applause, laughter, silence and other meaningful non-speech events.

### 3.3 Editorial history

Maintain at least the following short-term history:

- current programme shot and when it began;
- previous shots and their durations;
- when every person was last shown;
- recent questions, answers, references and reactions;
- unresolved conversational handoffs;
- recently shown objects or demonstrations;
- planned shot sequence, if one exists;
- pending decision and whether the renderer acknowledged it;
- operator overrides and the reason for them;
- invalidated evidence after reconnects or camera changes.

This history prevents repetitive cuts, jumpy direction and reactions that arrive
after the moment has already passed.

### 3.4 Production state

Track:

- renderer readiness and acknowledgement;
- programme mode: `MANUAL`, `ASSIST` or `AUTO`;
- manual HOLD and recent operator actions;
- stream epochs and renderer/control generations;
- camera health and approved safety shot;
- master-audio health;
- recording state;
- confidence thresholds and degraded capabilities.

Manual intent is authoritative. AUTO must never silently resume after a
reconnect, backend restart, HOLD or uncertain renderer state.

---

## 4. Multi-timescale architecture

Full context awareness does not require sending every video frame to an
expensive VLM. Use several cooperating loops.

### Fast loop: approximately 10–30 Hz

Use inexpensive local or streaming signals for:

- frame freshness and frozen-frame detection;
- person/face tracks;
- mouth motion and visual speech activity;
- movement and gesture onset;
- shot quality, occlusion and exposure;
- audio voice activity;
- renderer health and acknowledgements.

### Semantic loop: approximately 1–2 observations per second per camera

Use a VLM or equivalent visual-semantic model to produce structured changes:

- who is doing what;
- important reactions;
- object or demonstration visibility;
- entrances, exits and transitions;
- relationship between people across cameras;
- whether a shot is editorially useful.

Run this adaptively. Increase sampling when motion, speaker change, laughter,
applause, a script transition or uncertainty occurs. Reduce it during a stable
monologue.

### Language loop: streaming

Use streaming transcription plus a language model or deterministic dialogue
analysis to update:

- speaker turns;
- addressees and references;
- topic and dialogue act;
- anticipated response;
- narrative importance;
- optional script alignment.

Do not wait for a complete paragraph before recognizing an obvious handoff, but
do not cut on unstable interim transcription alone.

### Director loop: approximately 2–5 decisions per second

The director consumes the consolidated state, generates candidate shots, scores
them and chooses one of:

```text
HOLD_CURRENT
CUT_TO_CAMERA
SUGGEST_CAMERA
USE_SAFETY_SHOT
WAIT_FOR_MORE_EVIDENCE
```

The loop may reconsider frequently, but it should execute cuts only when the
editorial benefit exceeds the continuity cost.

---

## 5. Suggested processing pipeline

```text
Three live feeds ──► fast visual observers ─┐
                                            │
Master audio ─────► VAD + streaming ASR ────┼─► Temporal world-state reducer
                                            │
Optional script ──► script/segment tracker ─┘

World state ─► candidate generator ─► editorial scorer ─► continuity guard
            ─► ASSIST/AUTO gate ─► renderer command ─► ACK ─► history update
```

The VLM and language model must return structured observations. They must not
directly mutate the renderer. A deterministic control layer validates freshness,
mode, camera health, minimum duration and renderer generation before execution.

---

## 6. Structured observations

An illustrative camera observation:

```json
{
  "cameraId": "CAM-GUEST",
  "observedAtMs": 1789874000123,
  "streamEpoch": 4,
  "shotType": "MEDIUM_CLOSE_UP",
  "people": [
    {
      "trackId": "track-7",
      "identity": "Saerah",
      "identityConfidence": 0.94,
      "speakingProbability": 0.88,
      "attentionTarget": "HOST",
      "action": "answering thoughtfully",
      "reaction": "emotionally engaged",
      "visualQuality": 0.92
    }
  ],
  "objects": [],
  "motion": "LOW",
  "occluded": false,
  "healthy": true,
  "semanticConfidence": 0.9
}
```

An illustrative conversation state:

```json
{
  "currentSpeaker": "Saerah",
  "dialogueAct": "PERSONAL_ANSWER",
  "topic": "motivation for building CUE",
  "expectedNextSpeaker": "HOST",
  "addressedPeople": ["HOST"],
  "emotionalSalience": 0.87,
  "overlap": false,
  "sentenceLikelyComplete": false,
  "scriptBeat": "GUEST_ORIGIN_STORY",
  "scriptAlignmentConfidence": 0.71
}
```

An illustrative director proposal:

```json
{
  "decision": "HOLD_CURRENT",
  "cameraId": "CAM-GUEST",
  "confidence": 0.91,
  "minimumHoldMs": 2800,
  "reasonCodes": [
    "ACTIVE_SPEAKER",
    "EMOTIONAL_MOMENT",
    "STRONG_COMPOSITION",
    "SENTENCE_CONTINUES"
  ],
  "alternatives": [
    {"cameraId": "CAM-WIDE", "score": 0.51},
    {"cameraId": "CAM-HOST", "score": 0.37}
  ],
  "stateVersion": 182,
  "streamEpoch": 4,
  "rendererGeneration": 3
}
```

Every executed or rejected decision should be explainable from reason codes and
the state version used.

---

## 7. Editorial decision model

Candidate scoring can begin with this conceptual model:

```text
shot_score =
    narrative_relevance
  + active_or_expected_speaker_value
  + reaction_value
  + demonstration_value
  + visual_quality
  + composition_value
  + variety_value
  + script_support
  - continuity_cost
  - rapid_cut_penalty
  - stale_observation_penalty
  - uncertainty_penalty
  - camera_health_penalty
```

These are not fixed global weights. The dialogue state changes their meaning:

- During a personal story, emotional continuity dominates reaction hunting.
- During a physical demonstration, object visibility dominates facial close-up.
- During overlapping debate, the wide shot gains value.
- During an entrance, spatial explanation gains value.
- During a punchline pause, a reaction becomes valuable for a short window.
- During uncertainty, holding a healthy current shot is preferable to guessing.

The system should generate a small shot plan when the next beat is predictable:

```text
wide entrance → guest close-up → host reaction → guest answer
```

The plan is provisional. Live evidence can replace it at any moment.

---

## 8. Continuity and cinematic rules

Use these as guardrails, not absolute creative laws:

- Avoid cutting for one-word acknowledgements such as “yeah” or “exactly.”
- Prefer a minimum shot duration of roughly 2–3 seconds unless safety requires
  an immediate change.
- Allow shorter 1–2 second reaction shots when the reaction is strong and the
  original speaker pauses.
- Do not show a reaction so late that it no longer belongs to the moment.
- Avoid repeatedly bouncing between two cameras during rapid conversation.
- Use a wide shot to establish entrances, exits, movement and group dynamics.
- Avoid cutting between nearly identical framings without narrative benefit.
- Stay on an emotionally important speaker through brief interruptions.
- Do not cut to a person solely because their name appears in the transcript.
- Prefer HOLD when the candidate advantage is small.
- A failed or stale camera can never win regardless of semantic relevance.
- Renderer acknowledgement is required before history records a successful cut.

---

## 9. Diverse examples demonstrating real context awareness

### 1. Guest entrance

**Event:** The host says, “Let’s welcome Saerah to the stage.” Saerah is still
walking in.

**Good direction:** Show `CAM-WIDE` for the entrance and spatial movement. Cut
to `CAM-GUEST` after she reaches her position and the shot is composed.

**Bad rigid behaviour:** Immediately show an empty guest chair because the name
“Saerah” was detected.

### 2. Directed question without a name

**Event:** The host turns toward Daniel and says, “What do you think?”

**Good direction:** Infer the addressee from gaze, body orientation, recent
conversation and camera visibility. Anticipate Daniel’s response.

**Power shown:** Addressee understanding without keyword dependence.

### 3. Name mentioned, but no cut

**Event:** Daniel says, “Earlier, Saerah made a great point about accessibility,”
then continues developing his own argument.

**Good direction:** Stay on Daniel. Saerah is the topic, not the visual owner of
the moment. A brief reaction is allowed only if she visibly reacts and Daniel
pauses.

**Power shown:** Reference resolution and narrative ownership.

### 4. Joke and reaction

**Event:** Saerah delivers a punchline. Daniel laughs visibly while Saerah pauses.

**Good direction:** Use a short Daniel or wide reaction shot, then return before
Saerah continues.

**Power shown:** Visual reaction timing across cameras.

### 5. Emotionally important answer

**Event:** Saerah describes a difficult personal experience. The host briefly
says “mm-hmm.”

**Good direction:** Remain on Saerah. The acknowledgement is not a speaker
handoff, and the emotional moment should not be interrupted.

**Power shown:** Emotional salience and back-channel understanding.

### 6. Unscripted speaker transition

**Event:** The host finishes a question. Daniel leans forward, inhales and raises
his hand before speaking.

**Good direction:** Prepare Daniel as the next candidate and cut as he takes the
floor, even though nobody names him.

**Power shown:** Anticipation from multimodal cues.

### 7. Overlapping debate

**Event:** Two guests disagree and briefly speak over one another.

**Good direction:** Use `CAM-WIDE` while ownership is ambiguous. Once one person
clearly establishes the floor, move to that person and hold.

**Power shown:** Uncertainty-aware group coverage rather than rapid oscillation.

### 8. Physical demonstration

**Event:** A speaker says, “Here is how the prototype works,” reaches for a
device and begins demonstrating it.

**Good direction:** Select whichever healthy camera shows the device, hands and
speaker relationship most clearly. Face prominence is secondary.

**Power shown:** Understanding the visual subject of the story.

### 9. Silent visual event

**Event:** A prop falls over while another person is speaking.

**Good direction:** If the event is harmless and relevant, briefly use the wide
shot; if it is distracting and unrelated, remain on the speaker.

**Power shown:** Visual relevance judgment rather than motion chasing.

### 10. “Both of you”

**Event:** The host asks, “How would both of you approach this differently?”

**Good direction:** Begin with the wide/group view to establish the shared
question, then follow the first respondent.

**Power shown:** Plural addressee and conversational structure.

### 11. Host listening versus guest speaking

**Event:** A guest explains a complex idea while the host produces a thoughtful,
visually strong reaction.

**Good direction:** Mostly remain with the guest; take the host reaction only at
a natural sentence boundary, then return.

**Power shown:** Reaction value balanced against speaker continuity.

### 12. Segment transition

**Event:** “That brings us to our final question.”

**Good direction:** Use a wide reset or host shot that visually communicates a
new segment, then select the question recipient.

**Power shown:** Discourse structure rather than literal name matching.

### 13. Applause

**Event:** The audience or group applauds after an announcement.

**Good direction:** Use a wide shot for the collective moment. If the honoured
person has a strong reaction, take a brief close-up before returning to the host.

**Power shown:** Non-speech event and social meaning.

### 14. Speaker leaves their frame

**Event:** The active guest stands and walks toward a display, leaving the guest
close-up.

**Good direction:** Transition to the wide shot before the close-up becomes
empty, then choose the best demonstration view.

**Power shown:** Motion prediction and proactive camera selection.

### 15. Camera obstruction

**Event:** The ideal guest camera is covered or freezes during an answer.

**Good direction:** Preserve master audio and select the best healthy contextual
alternative—usually wide, sometimes the listening host. Do not cut back until
the guest camera has been stable long enough.

**Power shown:** Context preserved during technical failure.

### 16. Unexpected interruption

**Event:** An off-screen organizer interrupts with an important announcement.

**Good direction:** Use wide if it explains where attention shifted; otherwise
hold a neutral listening shot. Do not hunt randomly for a face that is not on a
camera.

**Power shown:** Honest handling of an unseen source.

### 17. Classroom or workshop

**Event:** A teacher asks a question, several participants look toward one
person, and that person starts answering.

**Good direction:** Use social attention plus voice onset to locate the answer,
then show the relevant diagram when it becomes the subject.

**Power shown:** Group gaze, speaker discovery and object-aware coverage.

### 18. Awards or announcement

**Event:** The host slowly reveals the winner while cameras show nominees.

**Good direction:** Hold suspense on the host, then show the winner’s genuine
reaction, followed by wide movement toward the stage.

**Power shown:** Temporal storytelling across multiple beats.

### 19. Podcast-style interruption

**Event:** A guest starts to interject but stops and lets the current speaker
continue.

**Good direction:** Do not cut merely because mouth motion began. Preserve the
current speaker and clear the pending handoff.

**Power shown:** Intent recognition and aborted-turn handling.

### 20. End of segment

**Event:** “Thank you both. We’ll be right back.” Guests relax and begin moving.

**Good direction:** Return to the host for the closing line, then use a stable
wide shot or approved slate. Do not keep chasing informal post-segment speech.

**Power shown:** Programme-state awareness.

---

## 10. ASSIST, AUTO and manual control

### MANUAL

The producer selects shots. CUE may show context and health but must not execute
semantic decisions. Safety failover may remain available if explicitly approved.

### ASSIST

CUE proposes the best shot with a short explanation:

```text
Suggested: CAM-WIDE
Reason: two speakers overlapping; group relationship is important
Confidence: 0.78
```

The operator accepts or rejects the proposal. Rejections become evaluation data
but must not immediately retrain a model during the live event.

### AUTO

CUE executes only when:

- the complete current system has passed its gates;
- renderer state and acknowledgement are authoritative;
- evidence is fresh for the current stream epoch and generation;
- the selected camera is healthy;
- decision confidence exceeds the context-specific threshold;
- continuity and minimum-duration constraints allow the cut;
- no manual HOLD or recent override blocks it.

Low confidence should fall back to HOLD, a safe wide shot or an ASSIST
suggestion—not a random cut.

---

## 11. Failure and uncertainty behaviour

| Condition | Required response |
|---|---|
| One feed stale or frozen | Remove it from candidates; use a healthy contextual alternative |
| All semantic models unavailable | Preserve manual switching and camera-health failover |
| Transcript uncertain | Rely more on visual/temporal evidence; prefer HOLD over guessing |
| Visual identity uncertain | Use role/position/track descriptions; never assert a name |
| Conflicting audio and visual speaker evidence | Wait briefly or use wide; record ambiguity |
| Late model response | Reject it using state version/epoch/generation |
| Renderer ACK missing | Do not claim the cut succeeded; degrade to ASSIST/MANUAL |
| Backend or browser reconnect | Invalidate stale state and return to ASSIST |
| Master audio missing | Alert immediately; do not silently select another laptop mic |
| Manual operator action | Cancel or supersede incompatible pending AI decisions |

The programme should remain usable without the AI director.

---

## 12. Latency targets

Use these as design targets to test on the actual machines and network, not as
unverified claims:

| Event | Target |
|---|---:|
| Camera-health failure to safe output | under 1.5 seconds |
| Clear speaker handoff to proposed shot | approximately 500–1000 ms |
| Renderer command to visible picture | measure p50/p95; target under 300 ms |
| Reaction detection window | useful within approximately 1 second |
| Manual TAKE | always faster and more authoritative than AI |

Do not optimize latency by removing temporal reasoning. A wrong cut in 100 ms
is worse than a correct hold followed by a cut at 700 ms.

---

## 13. Evaluation scenarios

Build deterministic recorded fixtures and then repeat them with the real
four-laptop system.

Minimum scenario set:

1. name mentioned but current speaker should remain on-air;
2. directed question with no name;
3. one-word back-channel that must not cause a cut;
4. punchline plus visible reaction;
5. emotional answer plus host acknowledgement;
6. overlapping speech resolved through wide;
7. aborted interruption;
8. object demonstration;
9. entrance and exit;
10. active camera failure;
11. stale VLM decision arriving after a newer state;
12. reconnect returning to ASSIST;
13. operator override defeating a pending AUTO cut;
14. script deviation;
15. unknown identity with role-based fallback.

For each scenario record:

- expected acceptable shots, not just one brittle exact sequence;
- unacceptable shots;
- decision latency;
- executed and rejected reason codes;
- shot duration and rapid-cut count;
- camera health at decision time;
- model confidence;
- renderer acknowledgement;
- operator acceptance or correction.

Useful aggregate metrics include:

- editorial acceptance rate;
- wrong-person cut rate;
- missed-reaction rate;
- cuts per minute;
- rapid-cut violations;
- stale decision rejection rate;
- unhealthy-camera selection rate, which must be zero;
- p50/p95 decision-to-picture latency;
- percentage of time the system correctly chose to hold.

“Hold” is a valid and often excellent editorial decision. Do not evaluate the
system only by how often it changes cameras.

---

## 14. Suggested 90-second showcase

Use a rehearsed structure but allow natural wording so the system cannot rely on
exact phrases.

1. **Entrance:** Host welcomes Saerah; CUE uses wide, then guest.
2. **Unnamed handoff:** Host looks at Saerah and asks, “What inspired this?”
3. **Emotional continuity:** Host says “mm-hmm”; CUE stays on Saerah.
4. **Reference trap:** Host mentions Saerah while continuing his own point; CUE
   stays on the host.
5. **Demonstration:** Saerah reaches for the prototype; CUE chooses the view that
   shows the action.
6. **Reaction:** A joke produces a visible reaction; CUE briefly captures it.
7. **Overlap:** Two people begin speaking; CUE uses wide instead of oscillating.
8. **Camera failure:** Temporarily obstruct the ideal camera; CUE keeps audio and
   selects a healthy alternative.
9. **Recovery:** The camera returns; CUE waits for stability rather than cutting
   immediately.
10. **Closing transition:** Host ends the segment; CUE returns to host then wide.

Display a small “Director reasoning” panel during development or judging:

```text
ON AIR: CAM-WIDE
Reason: overlapping speakers + both visible
Held for: 3.2 s
Alternative: CAM-GUEST (0.62)
Mode: ASSIST
```

This makes the contextual intelligence visible without cluttering the final
programme recording.

---

## 15. Privacy, consent and identity

- Obtain consent before recording or using face references.
- Prefer ephemeral, in-memory identity state for the event.
- A restart requires re-enrolment unless explicit short-lived persistence is
  approved.
- Use role/track descriptions when identity confidence is insufficient.
- Never expose face embeddings, secrets, provider tokens or raw private evidence
  in logs or Git.
- Do not infer sensitive personal traits or diagnose emotion. “Visible reaction”
  is an editorial observation, not a psychological claim.
- Provide manual deletion and event-end cleanup.

The director must work in role-based mode without named identity.

---

## 16. Implementation sequence

### Phase A — temporal state and explainable rule baseline

- Consolidate transcript turns, camera health and renderer history.
- Implement dialogue acts, addressed-person inference and back-channel handling.
- Produce scored suggestions with reason codes.
- Add minimum-duration, HOLD and stale-decision guards.

### Phase B — multi-camera visual context

- Add per-camera person tracks, composition and health.
- Detect movement, gestures, entrances/exits and object visibility.
- Maintain observations over time instead of frame captions.

### Phase C — multimodal world model

- Fuse transcript, visual observations, script alignment and shot history.
- Add reactions, demonstration state and anticipated handoffs.
- Use adaptive VLM sampling around meaningful changes.

### Phase D — editorial planner

- Generate short provisional shot sequences.
- Add context-dependent scoring and confidence calibration.
- Evaluate acceptable alternatives instead of one exact “correct” camera.

### Phase E — guarded live execution

- Run in ASSIST first.
- Record operator acceptance/corrections.
- Enable AUTO only after real hardware, reconnect, latency and failure gates pass.
- Preserve immediate manual control and an honest degraded mode.

---

## 17. Definition of done

The context-aware director is not complete merely because it switches among
three feeds. It is complete only when:

- all cameras are observed continuously enough to understand meaningful change;
- decisions use persistent multimodal context and recent shot history;
- named references are separated from visual ownership;
- addressees and expected speakers can be inferred without names;
- reactions, movement, demonstrations and group moments affect direction;
- uncertainty often results in a justified hold or wide shot;
- visual health can override semantic preference;
- stale decisions cannot reach the renderer;
- manual intent always wins;
- master audio remains continuous across cuts;
- every decision has compact reason codes and traceable state;
- deterministic fixtures and real multi-laptop trials pass;
- the system degrades honestly when identity, transcription, VLM or network
  services fail.

The essential test is this:

> If all names were removed from the transcript, could CUE still direct a
> coherent, watchable programme by understanding who is acting, speaking,
> reacting, demonstrating and transitioning?

If the answer is no, the implementation is still a trigger-based switcher and
does not satisfy this specification.
