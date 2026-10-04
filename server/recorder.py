"""Program recorder: what the audience saw, the master audio, and a cut log.

- video.mp4   : H.264 in a fragmented MP4 written through an ffmpeg pipe, so the file stays
                playable even if the server dies mid-show (falls back to cv2.VideoWriter when
                ffmpeg is missing). Every tick writes the live camera's latest frame
                (letterboxed to the recording size); SLATE ticks write a card.
- audio.wav   : master mic PCM (16 kHz int16 mono) appended continuously. Video cuts
                never touch this stream: the programme audio stays one continuous take.
- cuts.jsonl  : one line per program change (offset, camera, evidence, reason, cue, latency).
- transcript.jsonl : clauses and utterance ends as they finalise.
- summary.json: status, duration, frame/cut counts, final file names.
On stop, video and audio are muxed into program.mp4 with ffmpeg when it is installed;
otherwise both files are kept side by side and the summary says so.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("cue.recorder")


def _ffmpeg_h264_encoder() -> str | None:
    """libx264 if this ffmpeg has it, else the macOS hardware encoder, else None."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    try:
        out = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for enc in ("libx264", "h264_videotoolbox"):
        if f" {enc} " in out:
            return enc
    return None


class _FfmpegWriter:
    """Raw BGR frames -> ffmpeg stdin -> fragmented H.264 MP4 (playable while being written)."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float, encoder: str):
        w, h = size
        cmd = [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "pipe:0",
               "-an", "-c:v", encoder]
        if encoder == "libx264":
            cmd += ["-preset", "veryfast", "-crf", "23", "-tune", "zerolatency"]
        else:
            cmd += ["-b:v", "1500k", "-realtime", "1"]
        cmd += ["-pix_fmt", "yuv420p", "-g", str(int(fps * 2)),
                "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-flush_packets", "1", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.encoder = encoder
        self.failed = False
        self.error = ""

    def isOpened(self) -> bool:
        return self.proc.poll() is None and not self.failed

    def write(self, frame: np.ndarray) -> None:
        if self.failed or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as e:
            self.failed = True
            try:
                self.error = (self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else "")[:400]
            except Exception:
                pass
            log.error("ffmpeg writer died: %s %s", e, self.error)

    def release(self) -> None:
        if self.failed:
            try:
                self.proc.kill()
            except Exception:
                pass
            return
        try:
            _, err = self.proc.communicate(timeout=30)  # closes stdin, ffmpeg finalises the last fragment
            if err:
                self.error = err.decode("utf-8", "replace")[:400]
                if self.proc.returncode not in (0, None):
                    log.warning("ffmpeg writer exit %s: %s", self.proc.returncode, self.error)
        except Exception as e:
            log.warning("ffmpeg writer did not exit cleanly: %s", e)
            self.proc.kill()


class Recorder(threading.Thread):
    def __init__(self, out_dir: Path, get_program: ProgramGetter, fps: float = 15.0,
                 size: tuple[int, int] = (640, 360), sample_rate: int = 16000):
        super().__init__(daemon=True, name="cue-recorder")
        self.out_dir = out_dir
        self.get_program = get_program
        self.fps = fps
        self.size = size
        self.sample_rate = sample_rate
        self.started_wall: float | None = None
        self.started_mono: float | None = None
        self._stop_ev = threading.Event()
        self._lock = threading.Lock()
        self._writer = None            # _FfmpegWriter or cv2.VideoWriter
        self._writer_lock = threading.Lock()   # writer lifetime only (run() vs stop())
        self.video_codec = ""
        self.audio_peak = 0.0          # running peak of |sample| in 0..1, for a 'mic silent' warning
        self.io_error = ""
        self._wav: wave.Wave_write | None = None
        self._cuts_f = None
        self._transcript_f = None
        self.frames = 0
        self.audio_bytes = 0
        self.audio_pad_s = 0.0      # silence inserted so audio lines up with video when the mic starts late
        self._audio_started = False
        self.cuts = 0
        self.summary: dict = {}
        self.finished = False

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        if not (1 <= self.fps <= 60):
            raise RuntimeError(f"recording fps {self.fps} out of range 1-60 (CUE_RECORD_FPS)")
        enc = _ffmpeg_h264_encoder()
        if enc:
            self._writer = _FfmpegWriter(self.out_dir / "video.mp4", self.size, self.fps, enc)
            self.video_codec = f"h264/{enc} fragmented mp4"
        else:
            for fourcc in ("avc1", "mp4v"):
                w = cv2.VideoWriter(str(self.out_dir / "video.mp4"), cv2.VideoWriter_fourcc(*fourcc), self.fps, self.size)
                if w.isOpened():
                    self._writer = w
                    self.video_codec = f"cv2/{fourcc} (no ffmpeg: file is finalised only on stop)"
                    break
            else:
                raise RuntimeError("no video writer available (ffmpeg missing and cv2.VideoWriter failed)")
        self._wav = wave.open(str(self.out_dir / "audio.wav"), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.sample_rate)
        self._cuts_f = open(self.out_dir / "cuts.jsonl", "w", encoding="utf-8")
        self._transcript_f = open(self.out_dir / "transcript.jsonl", "w", encoding="utf-8")
        self._write_summary("recording")
        super().start()
        log.info("recording to %s (%dx%d @ %.0f fps)", self.out_dir, self.size[0], self.size[1], self.fps)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_mono if self.started_mono else 0.0

    def stop(self) -> dict:
        self._stop_ev.set()
        if self.is_alive():
            self.join(timeout=5)
        with self._writer_lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
        with self._lock:
            if self._wav is not None:
                self._wav.close()
                self._wav = None
            for f in (self._cuts_f, self._transcript_f):
                if f is not None:
                    f.close()
            self._cuts_f = self._transcript_f = None
        audio_s = self.audio_bytes / (2 * self.sample_rate)
        final = "video.mp4"
        muxed = False
        mux_error = ""
        if audio_s > 0.5 and shutil.which("ffmpeg"):
            try:
                # Pad audio to the video length and cut at the known video duration: '-shortest' with
                # 'apad' can run away on some ffmpeg builds, an explicit -t never does.
                video_s = self.frames / self.fps
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.out_dir / "video.mp4"),
                     "-i", str(self.out_dir / "audio.wav"), "-c:v", "copy", "-c:a", "aac", "-af", "apad",
                     "-t", f"{video_s:.3f}", str(self.out_dir / "program.mp4")],
                    check=True, timeout=120, capture_output=True, text=True,
                )
                muxed = True
                final = "program.mp4"
            except Exception as e:  # keep both files; never lose the recording
                mux_error = f"{type(e).__name__}: {getattr(e, 'stderr', '') or e}"[:300]
                log.warning("ffmpeg mux failed, keeping video.mp4 + audio.wav: %s", mux_error)
        self.finished = True
        self._write_summary("done", final=final, muxed=muxed, audio_s=round(audio_s, 2), mux_error=mux_error,
                            video_codec=self.video_codec, io_error=self.io_error, audio_peak=round(self.audio_peak, 4))
        log.info("recording stopped: %s (%.1fs, %d frames, %d cuts, audio %.1fs, muxed=%s)",
                 self.out_dir / final, self.elapsed(), self.frames, self.cuts, audio_s, muxed)
        return self.summary

    def _write_summary(self, status: str, **extra) -> None:
        self.summary = {
            "dir": str(self.out_dir), "status": status, "started_wall": self.started_wall,
            "duration_s": round(self.elapsed(), 2), "fps": self.fps, "size": list(self.size),
            "frames": self.frames, "cuts": self.cuts, "audio_s": round(self.audio_bytes / (2 * self.sample_rate), 2),
            "audio_pad_s": self.audio_pad_s,
            "ffmpeg": bool(shutil.which("ffmpeg")), **extra,
        }
        try:
            (self.out_dir / "summary.json").write_text(json.dumps(self.summary, indent=2))
        except Exception as e:
            log.warning("summary write failed: %s", e)

    # ----------------------------------------------------------------- inputs
    def audio(self, pcm: bytes) -> None:
        """Master audio tap. Called from the asyncio loop thread for every mic chunk.
        The mic may open 0.3-2 s after the video starts (CoreAudio init); the first chunk is
        preceded by that much silence so the track stays aligned with the video timeline."""
        with self._lock:
            if self._wav is None:
                return
            if not self._audio_started:
                self._audio_started = True
                gap_s = self.elapsed() - len(pcm) / (2 * self.sample_rate)
                if gap_s > 0.05:
                    pad = b"\x00" * (int(gap_s * self.sample_rate) * 2)
                    self._wav.writeframes(pad)
                    self.audio_bytes += len(pad)
                    self.audio_pad_s = round(gap_s, 3)
            self._wav.writeframes(pcm)
            self.audio_bytes += len(pcm)
            try:
                a = np.frombuffer(pcm, dtype=np.int16)
                if a.size:
                    self.audio_peak = max(self.audio_peak * 0.999, float(np.abs(a).max()) / 32768.0)
            except Exception:
                pass

    def log_cut(self, camera_id: str | None, evidence: str, reason: str, cue: dict | None, latency_ms: float | None) -> None:
        rec = {"t": round(self.elapsed(), 3), "wall": time.time(), "camera": camera_id, "evidence": evidence,
               "reason": reason, "cue": cue, "clause_to_cut_ms": None if latency_ms is None else round(latency_ms, 1)}
        with self._lock:
            if self._cuts_f is not None:
                try:
                    self._cuts_f.write(json.dumps(rec) + "\n")
                    self._cuts_f.flush()
                except OSError as e:  # disk full / volume gone: keep the show running, remember the error
                    self.io_error = self.io_error or f"cuts.jsonl: {e}"[:200]
                self.cuts += 1

    def log_transcript(self, utterance_id: str, text: str, final: bool, source: str) -> None:
        rec = {"t": round(self.elapsed(), 3), "wall": time.time(), "utterance_id": utterance_id, "text": text,
               "final": final, "source": source}
        with self._lock:
            if self._transcript_f is not None:
                try:
                    self._transcript_f.write(json.dumps(rec) + "\n")
                    self._transcript_f.flush()
                except OSError as e:
                    self.io_error = self.io_error or f"transcript.jsonl: {e}"[:200]

    # ------------------------------------------------------------------ video
    def run(self) -> None:
        interval = 1.0 / self.fps
        next_t = time.monotonic()
        last_key: tuple[str | None, int] | None = None
        last_frame = self._slate(None)
        while not self._stop_ev.is_set():
            cam_id, jpeg, seq = self.get_program()
            if cam_id is None or jpeg is None:
                frame = self._slate(cam_id)
                last_key = None
            elif last_key != (cam_id, seq):
                img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                frame = self._fit(img) if img is not None else last_frame
                last_key = (cam_id, seq)
            else:
                frame = last_frame
            last_frame = frame
            self._write_frame(frame)
            if self.frames % int(self.fps * 10) == 0:
                self._write_summary("recording")
            next_t += interval
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop_ev.wait(delay)
            elif delay < -1.0:
                # The thread stalled (sleep, GIL starvation). Never drop time: repeat the last frame
                # so frames/fps keeps tracking the wall clock, audio and cut offsets stay aligned.
                missing = min(int(-delay * self.fps), int(self.fps * 10))
                for _ in range(missing):
                    self._write_frame(frame)
                next_t = time.monotonic()

    def _write_frame(self, frame: np.ndarray) -> None:
        with self._writer_lock:
            w = self._writer
            if w is None:
                return
            w.write(frame)
            self.frames += 1

    def _fit(self, img: np.ndarray) -> np.ndarray:
        W, H = self.size
        h, w = img.shape[:2]
        if (w, h) == (W, H):
            return img
        s = min(W / w, H / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        y0, x0 = (H - nh) // 2, (W - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    def _slate(self, cam_id: str | None) -> np.ndarray:
        W, H = self.size
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        text = "SLATE" if cam_id is None else f"{cam_id}: no signal"
        scale = W / 640
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.6 * scale, 2)
        cv2.putText(canvas, text, ((W - tw) // 2, (H + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6 * scale, (90, 90, 110), 2, cv2.LINE_AA)
        return canvas


def list_recordings(root: Path) -> list[dict]:
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        sj = d / "summary.json"
        if d.is_dir() and sj.exists():
            try:
                s = json.loads(sj.read_text())
                s["name"] = d.name
                out.append(s)
            except Exception:
                continue
    return out
