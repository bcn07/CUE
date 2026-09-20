import time
import wave
from pathlib import Path

import cv2
import numpy as np

from server.recorder import Recorder, list_recordings

FX = Path(__file__).resolve().parent / "fixtures"


def _jpeg():
    img = cv2.imread(str(FX / "obama2.jpg"))
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (400, 500)))
    return buf.tobytes()


def test_recorder_writes_video_audio_cuts_and_aligns_late_mic(tmp_path):
    jpeg = _jpeg()
    state = {"cam": "B", "seq": 0}

    def getp():
        state["seq"] += 1
        return (state["cam"], jpeg, state["seq"]) if state["cam"] else (None, None, 0)

    r = Recorder(tmp_path / "take", getp, fps=15, size=(640, 360))
    r.start()
    time.sleep(0.8)                      # mic "opens" 0.8 s after the video started
    chunk = bytes(640)                   # 20 ms of silence at 16 kHz int16
    for _ in range(50):                  # 1.0 s of audio
        r.audio(chunk)
        time.sleep(0.02)
    r.log_cut("B", "identity", "fresh identity", {"action": "SHOW"}, 12.0)
    state["cam"] = None                  # SLATE for the tail
    time.sleep(0.4)
    s = r.stop()
    assert s["status"] == "done" and s["frames"] >= 25 and s["cuts"] == 1
    assert 0.6 <= s["audio_pad_s"] <= 1.1, s            # leading silence covers the late mic start
    with wave.open(str(tmp_path / "take" / "audio.wav")) as w:
        audio_s = w.getnframes() / w.getframerate()
    assert abs(audio_s - (s["audio_pad_s"] + 1.0)) < 0.1
    files = {p.name for p in (tmp_path / "take").iterdir()}
    assert {"video.mp4", "audio.wav", "cuts.jsonl", "transcript.jsonl", "summary.json"} <= files
    if s["muxed"]:
        cap = cv2.VideoCapture(str(tmp_path / "take" / "program.mp4"))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 640
        # the mux never truncates the video to a shorter audio track
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= s["frames"] - 3
    listing = list_recordings(tmp_path)
    assert listing and listing[0]["name"] == "take" and listing[0]["cuts"] == 1


def test_recorder_without_audio_keeps_video_only(tmp_path):
    jpeg = _jpeg()
    r = Recorder(tmp_path / "silent", lambda: ("A", jpeg, 1), fps=15, size=(320, 180))
    r.start()
    time.sleep(0.5)
    s = r.stop()
    assert s["muxed"] is False and s["final"] == "video.mp4" and s["audio_s"] == 0
    import subprocess
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                            str(tmp_path / "silent" / "video.mp4")], capture_output=True, text=True)
    assert "h264" in probe.stdout or "mpeg4" in probe.stdout, probe.stderr


def test_video_is_playable_while_still_recording(tmp_path):
    """A fragmented MP4 survives a crash: the file is readable before stop() is ever called."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        return
    jpeg = _jpeg()
    r = Recorder(tmp_path / "live", lambda: ("A", jpeg, int(time.time() * 100)), fps=15, size=(320, 180))
    r.start()
    time.sleep(3.0)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name:format=duration", "-of", "csv=p=0",
                            str(tmp_path / "live" / "video.mp4")], capture_output=True, text=True)
    r.stop()
    assert "h264" in probe.stdout, probe.stderr
    duration = float(probe.stdout.strip().splitlines()[-1])
    assert duration >= 1.5, probe.stdout
