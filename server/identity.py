"""Face identity, fully local: YuNet (detect) + SFace (128-d embedding), OpenCV only.

Measured on this MacBook: YuNet 640x360 ~5 ms, SFace ~5-10 ms per face.
Thresholds follow B's cue_vision defaults: cosine accept 0.363, margin 0.06
over the runner-up, and K consistent observations before an identity counts.
An unknown or ambiguous face stays unknown; the director then uses the fixed
mapping or WIDE instead of guessing.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("cue.identity")
try:  # OpenCV 5 prints a harmless DNN backend warning on model load; keep the console clean.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
MODEL_URLS = {
    YUNET: "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    SFACE: "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


MIN_BYTES = {YUNET: 200_000, SFACE: 38_000_000}  # real files: 232,589 and 38,696,353 bytes


def models_present(models_dir: Path) -> bool:
    """Both ONNX files exist and are complete (a git-lfs pointer is ~130 bytes; a cut-off
    download is anything smaller than the real file)."""
    for name in (YUNET, SFACE):
        p = models_dir / name
        if not p.exists() or p.stat().st_size < MIN_BYTES[name]:
            return False
    return True


@dataclass
class Face:
    box: tuple[int, int, int, int]
    score: float
    row: np.ndarray


@dataclass
class Observation:
    person_id: str | None
    similarity: float
    margin: float | None
    box: tuple[int, int, int, int]
    decision: str  # CANDIDATE | AMBIGUOUS | UNKNOWN | EMPTY


class FaceEngine:
    def __init__(self, models_dir: Path, score_thresh: float = 0.7, nms: float = 0.3, max_side: int = 720):
        self._lock = threading.Lock()
        self.det = cv2.FaceDetectorYN.create(str(models_dir / YUNET), "", (320, 320), score_thresh, nms, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(models_dir / SFACE), "")
        self._size: tuple[int, int] | None = None
        self.max_side = max_side

    def prepare(self, img: np.ndarray) -> np.ndarray:
        """Downscale big frames so detection stays a few ms."""
        h, w = img.shape[:2]
        m = max(h, w)
        if m > self.max_side:
            s = self.max_side / m
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return img

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        with self._lock:
            if self._size != (w, h):
                self.det.setInputSize((w, h))
                self._size = (w, h)
            _, faces = self.det.detect(img)
        if faces is None:
            return []
        out = []
        for r in faces:
            x, y, bw, bh = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            out.append(Face((max(0, x), max(0, y), max(1, bw), max(1, bh)), float(r[14]), r))
        out.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
        return out

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        with self._lock:
            crop = self.rec.alignCrop(img, face.row)
            feat = self.rec.feature(crop)
        v = np.asarray(feat, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


class Gallery:
    def __init__(self, accept: float = 0.363, margin: float = 0.06):
        self.accept = accept
        self.margin = margin
        self.embeddings: dict[str, list[np.ndarray]] = {}
        self._lock = threading.Lock()

    def replace(self, embeddings: dict[str, list[np.ndarray]]) -> None:
        with self._lock:
            self.embeddings = {k: list(v) for k, v in embeddings.items() if v}

    @property
    def size(self) -> int:
        return sum(len(v) for v in self.embeddings.values())

    def match(self, emb: np.ndarray) -> tuple[str | None, float, float | None, str]:
        with self._lock:
            items = list(self.embeddings.items())
        if not items:
            return None, 0.0, None, "EMPTY"
        scored = sorted(((max(float(np.dot(e, emb)) for e in embs), pid) for pid, embs in items), reverse=True)
        best, pid = scored[0]
        runner = scored[1][0] if len(scored) > 1 else None
        margin = None if runner is None else best - runner
        if best < self.accept:
            return None, best, margin, "UNKNOWN"
        if margin is not None and margin < self.margin:
            return None, best, margin, "AMBIGUOUS"
        return pid, best, margin, "CANDIDATE"


def ensure_decodable(img_path: Path) -> bool:
    """Phones upload HEIC (often under a .jpg name); OpenCV cannot read it. Convert in place to a
    real JPEG with macOS `sips` (or Pillow + pillow-heif if present). Returns True when readable."""
    import shutil
    import subprocess
    if cv2.imread(str(img_path)) is not None:
        return True
    tmp = img_path.with_suffix(".converted.jpg")
    ok = False
    if shutil.which("sips"):
        r = subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "90", str(img_path), "--out", str(tmp)],
                           capture_output=True, text=True, timeout=60)
        ok = r.returncode == 0 and tmp.exists() and cv2.imread(str(tmp)) is not None
    if not ok:
        try:
            from PIL import Image
            try:
                import pillow_heif  # type: ignore
                pillow_heif.register_heif_opener()
            except Exception:
                pass
            Image.open(img_path).convert("RGB").save(tmp, "JPEG", quality=90)
            ok = cv2.imread(str(tmp)) is not None
        except Exception:
            ok = False
    if ok:
        target = img_path if img_path.suffix.lower() in (".jpg", ".jpeg") else img_path.with_suffix(".jpg")
        tmp.replace(target)
        if target != img_path and img_path.exists():
            img_path.unlink()
        log.info("converted %s to JPEG (was not decodable, probably HEIC)", img_path.name)
        return True
    tmp.unlink(missing_ok=True)
    return False


def enroll_people(engine: FaceEngine, people_dir: Path) -> tuple[dict[str, list[np.ndarray]], dict[str, dict]]:
    """people_dir/<person_id>/*.jpg -> one embedding per photo (largest face).
    Returns (embeddings, report[person_id] = {photos, faces, failed:[filenames]})."""
    embeddings: dict[str, list[np.ndarray]] = {}
    report: dict[str, dict] = {}
    if not people_dir.exists():
        return embeddings, report
    for pdir in sorted(p for p in people_dir.iterdir() if p.is_dir()):
        pid = pdir.name
        rep = {"photos": 0, "faces": 0, "failed": []}
        for img_path in sorted(pdir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            rep["photos"] += 1
            if not ensure_decodable(img_path):
                rep["failed"].append(img_path.name + " (unreadable: not JPEG/PNG, and conversion failed)")
                continue
            if img_path.suffix.lower() not in (".jpg", ".jpeg") and not img_path.exists():
                img_path = img_path.with_suffix(".jpg")
            img = cv2.imread(str(img_path))
            if img is None:
                rep["failed"].append(img_path.name)
                continue
            img = engine.prepare(img) if max(img.shape[:2]) > 1280 else img
            faces = engine.detect(img)
            if not faces:
                rep["failed"].append(img_path.name)
                continue
            emb = engine.embed(img, faces[0])
            embeddings.setdefault(pid, []).append(emb)
            rep["faces"] += 1
        report[pid] = rep
    return embeddings, report


def reference_thumbnail(img_path: Path, size: int = 256) -> bytes | None:
    """Small JPEG of the whole photo (for the setup page and the VLM fallback)."""
    img = cv2.imread(str(img_path))
    if img is None and ensure_decodable(img_path):
        img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    s = size / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


class PresenceTracker:
    """Per camera: who is confirmed present and how old that evidence is.
    An identity is confirmed after `confirmations` consistent observations
    within `window_s`. Evidence age = time since the last confirming frame."""

    def __init__(self, confirmations: int = 2, window_s: float = 1.5):
        self.k = max(1, confirmations)
        self.window_s = window_s
        self._hits: dict[str, list[float]] = {}
        self._confirmed_at: dict[str, float] = {}
        self._sim: dict[str, float] = {}
        self._source: dict[str, str] = {}
        self.faces: list[dict] = []
        self.updated_at: float = 0.0
        self.frame_size: tuple[int, int] = (0, 0)
        self._lock = threading.Lock()

    def observe(self, obs: list[Observation], now: float, frame_size: tuple[int, int], source: str = "sface") -> None:
        with self._lock:
            for o in obs:
                if not o.person_id:
                    continue
                hits = [t for t in self._hits.get(o.person_id, []) if now - t <= self.window_s]
                hits.append(now)
                self._hits[o.person_id] = hits
                if len(hits) >= self.k:
                    self._confirmed_at[o.person_id] = now
                    self._sim[o.person_id] = o.similarity
                    self._source[o.person_id] = source
            self.faces = [
                {"box": list(o.box), "person_id": o.person_id, "similarity": round(o.similarity, 3), "decision": o.decision}
                for o in obs
            ]
            self.frame_size = frame_size
            self.updated_at = now

    def observe_ids(self, ids: list[str], now: float, source: str, similarity: float = 0.0) -> None:
        """Identity evidence without boxes (VLM fallback)."""
        with self._lock:
            for pid in ids:
                self._confirmed_at[pid] = now
                self._sim[pid] = similarity
                self._source[pid] = source
            self.updated_at = now

    def confirmed(self, now: float, max_age_s: float) -> dict[str, float]:
        with self._lock:
            return {pid: now - t for pid, t in self._confirmed_at.items() if now - t <= max_age_s}

    def snapshot(self, now: float, max_age_s: float) -> dict:
        with self._lock:
            present = {
                pid: {"age_s": round(now - t, 2), "similarity": round(self._sim.get(pid, 0.0), 3), "source": self._source.get(pid, "")}
                for pid, t in self._confirmed_at.items() if now - t <= max_age_s
            }
            return {"faces": list(self.faces), "present": present, "frame_size": list(self.frame_size),
                    "age_s": round(now - self.updated_at, 2) if self.updated_at else None}


FrameGetter = Callable[[], dict[str, tuple[int, bytes]]]


class IdentityWorker(threading.Thread):
    """Background thread: latest JPEG per camera -> faces -> gallery match -> tracker."""

    def __init__(self, engine: FaceEngine, gallery: Gallery, trackers: dict[str, PresenceTracker],
                 get_frames: FrameGetter, interval_s: float = 0.15, max_faces: int = 6,
                 on_update: Callable[[str, list[Observation], np.ndarray | None], None] | None = None,
                 tracker_factory: Callable[[], PresenceTracker] | None = None):
        super().__init__(daemon=True, name="cue-identity")
        self.engine, self.gallery, self.trackers = engine, gallery, trackers
        self.get_frames = get_frames
        self.interval_s = interval_s
        self.max_faces = max_faces
        self.on_update = on_update
        self.tracker_factory = tracker_factory or PresenceTracker
        self._seen: dict[str, int] = {}
        self._stop_ev = threading.Event()
        self.stats = {"frames": 0, "faces": 0, "last_ms": 0.0, "avg_ms": 0.0}

    def stop(self) -> None:
        self._stop_ev.set()

    def run(self) -> None:
        while not self._stop_ev.is_set():
            t_loop = time.perf_counter()
            frames = self.get_frames()
            for cam_id, (seq, jpeg) in frames.items():
                if self._seen.get(cam_id) == seq:
                    continue
                self._seen[cam_id] = seq
                try:
                    self._process(cam_id, jpeg)
                except Exception as e:  # never let one bad frame kill identity
                    log.warning("identity error on %s: %s", cam_id, e)
            dt = time.perf_counter() - t_loop
            self._stop_ev.wait(max(0.01, self.interval_s - dt))

    def _process(self, cam_id: str, jpeg: bytes) -> None:
        t0 = time.perf_counter()
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        img = self.engine.prepare(img)
        h, w = img.shape[:2]
        faces = self.engine.detect(img)[: self.max_faces]
        obs: list[Observation] = []
        for f in faces:
            if f.box[2] < 0.06 * w:  # too small to identify; keep the box, no identity
                obs.append(Observation(None, 0.0, None, f.box, "TOO_SMALL"))
                continue
            emb = self.engine.embed(img, f)
            pid, sim, margin, decision = self.gallery.match(emb)
            obs.append(Observation(pid, sim, margin, f.box, decision))
        now = time.monotonic()
        tracker = self.trackers.get(cam_id)
        if tracker is None:
            tracker = self.trackers[cam_id] = self.tracker_factory()
        tracker.observe(obs, now, (w, h))
        ms = (time.perf_counter() - t0) * 1000
        self.stats["frames"] += 1
        self.stats["faces"] += len(faces)
        self.stats["last_ms"] = round(ms, 1)
        self.stats["avg_ms"] = round(self.stats["avg_ms"] * 0.9 + ms * 0.1, 1)
        if self.on_update:
            self.on_update(cam_id, obs, img)
