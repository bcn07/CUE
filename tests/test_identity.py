import shutil
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures"
MODELS = ROOT / "models"

pytestmark = pytest.mark.skipif(
    not (FX / "obama.jpg").exists() or not (MODELS / "face_recognition_sface_2021dec.onnx").exists()
    or (MODELS / "face_recognition_sface_2021dec.onnx").stat().st_size < 10_000,
    reason="fixture photos or face models missing",
)


@pytest.fixture(scope="module")
def engine():
    from server.identity import FaceEngine
    return FaceEngine(MODELS)


@pytest.fixture()
def people_dir(tmp_path):
    (tmp_path / "barack").mkdir()
    (tmp_path / "joe").mkdir()
    shutil.copy(FX / "obama.jpg", tmp_path / "barack" / "obama.jpg")
    shutil.copy(FX / "biden.jpg", tmp_path / "joe" / "biden.jpg")
    return tmp_path


def _embed(engine, path):
    import cv2
    img = engine.prepare(cv2.imread(str(path)))
    faces = engine.detect(img)
    assert faces, path
    return engine.embed(img, faces[0])


def test_enroll_and_match_same_person_not_other(engine, people_dir):
    from server.identity import Gallery, enroll_people
    embs, report = enroll_people(engine, people_dir)
    assert report["barack"]["faces"] == 1 and report["joe"]["faces"] == 1
    g = Gallery()
    g.replace(embs)
    pid, sim, margin, decision = g.match(_embed(engine, FX / "obama2.jpg"))
    assert decision == "CANDIDATE" and pid == "barack" and sim > 0.363 and margin > 0.06
    g2 = Gallery()
    g2.replace({"barack": embs["barack"]})
    pid, sim, _, decision = g2.match(_embed(engine, FX / "biden.jpg"))
    assert decision == "UNKNOWN" and pid is None and sim < 0.363


def test_empty_gallery_never_identifies(engine):
    from server.identity import Gallery
    g = Gallery()
    assert g.match(_embed(engine, FX / "obama.jpg"))[3] == "EMPTY"


def test_presence_tracker_needs_k_observations_and_ages_out():
    from server.identity import Observation, PresenceTracker
    tr = PresenceTracker(confirmations=2, window_s=1.5)
    o = Observation("barack", 0.7, 0.4, (0, 0, 10, 10), "CANDIDATE")
    tr.observe([o], now=10.0, frame_size=(640, 360))
    assert tr.confirmed(10.1, 4.0) == {}
    tr.observe([o], now=10.2, frame_size=(640, 360))
    assert "barack" in tr.confirmed(10.3, 4.0)
    assert abs(tr.confirmed(12.2, 4.0)["barack"] - 2.0) < 1e-6
    assert tr.confirmed(15.0, 4.0) == {}
    # observations too far apart never confirm
    tr2 = PresenceTracker(confirmations=2, window_s=1.5)
    tr2.observe([o], now=0.0, frame_size=(640, 360))
    tr2.observe([o], now=5.0, frame_size=(640, 360))
    assert tr2.confirmed(5.0, 4.0) == {}


def test_identity_worker_confirms_person_on_camera(engine, people_dir):
    import cv2
    from server.identity import Gallery, IdentityWorker, PresenceTracker, enroll_people
    embs, _ = enroll_people(engine, people_dir)
    g = Gallery()
    g.replace(embs)
    img = cv2.imread(str(FX / "obama2.jpg"))
    img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpeg = buf.tobytes()
    seq = {"n": 0}
    lock = threading.Lock()

    def frames():
        with lock:
            seq["n"] += 1
            return {"B": (seq["n"], jpeg)}

    trackers: dict[str, PresenceTracker] = {}
    updates = []
    w = IdentityWorker(engine, g, trackers, frames, interval_s=0.05, on_update=lambda cam, obs, im: updates.append(cam),
                       tracker_factory=lambda: PresenceTracker(2, 1.5))
    w.start()
    deadline = time.time() + 5
    while time.time() < deadline and "barack" not in trackers.get("B", PresenceTracker()).confirmed(time.monotonic(), 4.0):
        time.sleep(0.05)
    w.stop()
    w.join(timeout=2)
    conf = trackers["B"].confirmed(time.monotonic(), 4.0)
    assert "barack" in conf, (conf, w.stats)
    assert w.stats["frames"] >= 2 and w.stats["last_ms"] < 200
    snap = trackers["B"].snapshot(time.monotonic(), 4.0)
    assert snap["faces"] and snap["faces"][0]["person_id"] == "barack"
    assert updates and updates[0] == "B"
