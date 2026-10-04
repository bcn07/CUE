"""Settings for the CUE director. Everything comes from the
environment (loaded from this repo's .env, then the team repo's .env as a
fallback for the two provider keys). Keys are never printed."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
REPO_ENV = ROOT.parent / "hackmit_2026_cue" / ".env"


KEY_SOURCES: dict[str, str] = {}


def _load_env() -> None:
    load_dotenv(ROOT / ".env")
    for k in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "LIVEKIT_API_SECRET"):
        if os.environ.get(k, "").strip():
            KEY_SOURCES[k] = "environment or .env"
    if REPO_ENV.exists():
        # Fallback: the team repo's .env fills names that are still empty. Documented in README; logged at startup.
        before = {k: os.environ.get(k, "").strip() for k in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY")}
        load_dotenv(REPO_ENV, override=False)
        for k, v in before.items():
            if not v and os.environ.get(k, "").strip():
                KEY_SOURCES[k] = f"{REPO_ENV} (team repo fallback)"


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"
    static_dir: Path = ROOT / "server" / "static"

    openai_api_key: str = ""
    deepgram_api_key: str = ""

    llm_model: str = "gpt-4.1-mini"
    llm_base_url: str = ""          # e.g. http://127.0.0.1:11434/v1 for Ollama -> OpenAI-compatible chat completions
    llm_api_key: str = ""           # dummy for local servers
    llm_timeout_s: float = 3.0
    fast_path: bool = True

    vlm_enabled: bool = True
    vlm_model: str = "gpt-4.1-mini"
    vlm_min_interval_s: float = 4.0
    vlm_timeout_s: float = 4.0

    scene_enabled: bool = False         # CUE_SCENE=1 turns the background scene tagger on (a second model on the GPU and in RAM)
    scene_model: str = "qwen2.5vl:3b"   # a vision model on the local OpenAI-compatible server (Ollama)
    scene_base_url: str = ""            # defaults to the LLM base URL
    scene_interval_s: float = 20.0      # one camera per interval: slow on purpose, the interpreter shares the GPU
    scene_max_age_s: float = 600.0
    scene_max_width: int = 384
    scene_timeout_s: float = 60.0

    deepgram_model: str = "nova-3"      # or flux-general-en: Deepgram's turn-based model on the v2 endpoint
    flux_eot_threshold: float = 0.7     # Flux: end-of-turn confidence that ends a turn (0.5-0.9)
    flux_eot_timeout_ms: int = 3000     # Flux: silence that ends a turn regardless (backstop)
    mic_device: str | None = None
    sample_rate: int = 16000

    ident_interval_s: float = 0.15
    identity_max_age_s: float = 4.0
    identity_accept: float = 0.363
    identity_margin: float = 0.06
    identity_confirmations: int = 2

    public_url: str = ""                # e.g. https://<words>.trycloudflare.com (shown on /setup for the phones)
    livekit_url: str = ""               # wss://<project>.livekit.cloud; when set, phones publish WebRTC there instead of JPEG over /ingest
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_room: str = "cue-oneshot"
    livekit_fps: float = 10.0           # JPEG re-encode rate per camera into the director
    livekit_jpeg_quality: int = 70
    livekit_max_width: int = 960
    team_api: str = ""                  # e.g. http://127.0.0.1:8000 (the team's apps/api)
    team_event_id: str = ""
    team_producer_secret: str = ""
    team_camera_map: str = "A=CAM-WIDE,B=CAM-GUEST,C=CAM-HOST"
    team_resume_mode: str = "ASSIST"     # ASSIST = suggestions the operator confirms; AUTO = the one-shot cuts on air
    signup_enabled: bool = True         # CUE_SIGNUP_ENABLED=0 switches the poller off whatever else is set
    signup_endpoint: str = ""           # the desk pages' Apps Script /exec URL: a capability, never logged
    signup_event_id: str = ""           # only rows written with this eventId; empty = every row
    signup_poll_s: float = 5.0
    signup_source: str = ""             # "env" or "config.js": where the endpoint came from, for the log

    record_fps: float = 15.0
    record_size: tuple[int, int] = (640, 360)
    auto_record: bool = False
    record_audio: bool = True

    min_shot_s: float = 2.5
    cue_lifetime_s: float = 3.0
    camera_stale_s: float = 1.5
    ui_state_hz: float = 4.0

    extra: dict = field(default_factory=dict)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_team_bridge(self) -> bool:
        return bool(self.team_api and self.team_event_id and self.team_producer_secret)

    @property
    def has_livekit(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)

    @property
    def has_llm(self) -> bool:
        """An interpreter is available: OpenAI key, or a local OpenAI-compatible server."""
        return bool(self.openai_api_key) or bool(self.llm_base_url)

    @property
    def llm_provider(self) -> str:
        return "local" if self.llm_base_url else ("openai" if self.openai_api_key else "none")

    @property
    def has_deepgram(self) -> bool:
        return bool(self.deepgram_api_key)

    @property
    def has_scene(self) -> bool:
        return self.scene_enabled and bool(self.scene_base_url) and bool(self.scene_model)

    @property
    def has_signup(self) -> bool:
        return self.signup_enabled and bool(self.signup_endpoint)


def apply_signup_config(s: "Settings", config_js: Path) -> None:
    """The endpoint: env wins, else the desk pages' own config.js (so the sheet is configured once).
    The event id: env wins, else config.js's, so the director sees the rows the pages write."""
    from .signup_sync import read_config_js
    cfg = read_config_js(config_js)
    if s.signup_endpoint:
        s.signup_source = "env"
    elif cfg.get("endpoint"):
        s.signup_endpoint, s.signup_source = cfg["endpoint"], "config.js"
    if s.signup_endpoint and not s.signup_event_id:
        s.signup_event_id = cfg.get("eventId", "")


def load_settings() -> Settings:
    _load_env()
    env = os.environ
    s = Settings(
        host=env.get("CUE_HOST", "0.0.0.0"),
        data_dir=Path(env.get("CUE_DATA_DIR") or (ROOT / "data")),
        models_dir=Path(env.get("CUE_MODELS_DIR") or (ROOT / "models")),
        port=int(env.get("CUE_PORT", "8000")),
        openai_api_key=env.get("OPENAI_API_KEY", "").strip(),
        deepgram_api_key=env.get("DEEPGRAM_API_KEY", "").strip(),
        llm_model=env.get("CUE_LLM_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini",
        llm_base_url=env.get("CUE_LLM_BASE_URL", "").strip().rstrip("/"),
        llm_api_key=env.get("CUE_LLM_API_KEY", "").strip() or "local",
        llm_timeout_s=_float("CUE_LLM_TIMEOUT_S", 6.0 if env.get("CUE_LLM_BASE_URL", "").strip() else 3.0),
        fast_path=_flag("CUE_FAST_PATH", "1"),
        vlm_enabled=_flag("CUE_VLM", "1"),
        vlm_model=env.get("CUE_VLM_MODEL", "").strip() or env.get("CUE_LLM_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini",
        vlm_min_interval_s=_float("CUE_VLM_MIN_INTERVAL_S", 4.0),
        deepgram_model=env.get("CUE_DEEPGRAM_MODEL", "nova-3").strip() or "nova-3",
        flux_eot_threshold=_float("CUE_FLUX_EOT_THRESHOLD", 0.7),
        flux_eot_timeout_ms=int(_float("CUE_FLUX_EOT_TIMEOUT_MS", 3000)),
        mic_device=(env.get("CUE_MIC_DEVICE") or None),
        ident_interval_s=_float("CUE_IDENT_INTERVAL_S", 0.15),
        identity_max_age_s=_float("CUE_IDENTITY_MAX_AGE_S", 4.0),
        min_shot_s=_float("CUE_MIN_SHOT_S", 2.5),
        cue_lifetime_s=_float("CUE_CUE_LIFETIME_S", 3.0),
        camera_stale_s=_float("CUE_CAMERA_STALE_S", 1.5),
        record_fps=min(60.0, max(1.0, _float("CUE_RECORD_FPS", 15.0))),
        public_url=env.get("CUE_PUBLIC_URL", "").strip().rstrip("/"),
        livekit_url=env.get("LIVEKIT_URL", "").strip().rstrip("/"),
        livekit_api_key=env.get("LIVEKIT_API_KEY", "").strip(),
        livekit_api_secret=env.get("LIVEKIT_API_SECRET", "").strip(),
        livekit_room=env.get("LIVEKIT_ROOM", "cue-oneshot").strip() or "cue-oneshot",
        livekit_fps=_float("CUE_LIVEKIT_FPS", 10.0),
        livekit_jpeg_quality=int(_float("CUE_LIVEKIT_JPEG_QUALITY", 70)),
        livekit_max_width=int(_float("CUE_LIVEKIT_MAX_WIDTH", 960)),
        team_api=env.get("CUE_TEAM_API", "").strip().rstrip("/"),
        team_event_id=env.get("CUE_TEAM_EVENT_ID", "").strip(),
        team_producer_secret=(env.get("CUE_TEAM_PRODUCER_SECRET") or env.get("CUE_PRODUCER_SECRET") or "").strip(),
        team_camera_map=env.get("CUE_TEAM_CAMERA_MAP", "A=CAM-WIDE,B=CAM-GUEST,C=CAM-HOST").strip(),
        team_resume_mode=env.get("CUE_TEAM_RESUME_MODE", "ASSIST").strip().upper() or "ASSIST",
        auto_record=_flag("CUE_AUTO_RECORD", "0"),
        record_audio=_flag("CUE_RECORD_AUDIO", "1"),
        scene_enabled=_flag("CUE_SCENE", "0"),
        scene_model=env.get("CUE_SCENE_MODEL", "qwen2.5vl:3b").strip() or "qwen2.5vl:3b",
        scene_base_url=(env.get("CUE_SCENE_BASE_URL") or env.get("CUE_LLM_BASE_URL") or "").strip().rstrip("/"),
        scene_interval_s=_float("CUE_SCENE_INTERVAL_S", 20.0),
        scene_max_age_s=_float("CUE_SCENE_MAX_AGE_S", 600.0),
        scene_max_width=int(_float("CUE_SCENE_MAX_WIDTH", 384)),
        scene_timeout_s=_float("CUE_SCENE_TIMEOUT_S", 60.0),
        signup_enabled=_flag("CUE_SIGNUP_ENABLED", "1"),
        signup_endpoint=env.get("CUE_SIGNUP_ENDPOINT", "").strip(),
        signup_event_id=env.get("CUE_SIGNUP_EVENT_ID", "").strip(),
        signup_poll_s=_float("CUE_SIGNUP_POLL_S", 5.0),
    )
    apply_signup_config(s, s.static_dir / "config.js")
    try:
        w, h = (env.get("CUE_RECORD_SIZE") or "640x360").lower().split("x")
        s.record_size = (max(160, int(w)), max(90, int(h)))
    except ValueError:
        s.record_size = (640, 360)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    (s.data_dir / "people").mkdir(parents=True, exist_ok=True)
    (s.data_dir / "recordings").mkdir(parents=True, exist_ok=True)
    return s
