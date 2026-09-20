"""Fast visual loop (spec §4): cheap per-frame signals per camera, computed inside the identity
worker on the frames it already decodes (~7 Hz per camera). Produces CameraObservation.

Signals: face tracks (IoU association), visual speech activity from mouth-region motion,
head yaw from the five YuNet landmarks, local and global motion, frozen frames, sharpness,
brightness, shot type from face size, entrances and exits."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .worldstate import CameraObservation, PersonObs


@dataclass
class _Track:
    track_id: str
    box: tuple[int, int, int, int]
    first_seen: float
    last_seen: float
    speaking: float = 0.0
    motion: float = 0.0
    yaw: float = 0.0
    mouth_prev: np.ndarray | None = None
    person_id: str | None = None
    id_conf: float = 0.0
    misses: int = 0


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class CameraObserver:
    def __init__(self, cam_id: str):
        self.cam_id = cam_id
        self.tracks: dict[str, _Track] = {}
        self._next = 1
        self._prev_gray: np.ndarray | None = None
        self._prev_hash: int | None = None
        self._same_frames = 0

    def observe(self, img: np.ndarray, faces: list, matches: list[tuple[str | None, float]], now: float) -> CameraObservation:
        """faces: identity.Face list (box, score, row with 5 landmarks); matches: (person_id, similarity) per face."""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
        # frozen: identical downscaled frames several times in a row (a stuck webcam or a still image)
        fh = hash(small.tobytes())
        self._same_frames = self._same_frames + 1 if fh == self._prev_hash else 0
        self._prev_hash = fh
        frozen = self._same_frames >= 8
        # global motion
        motion = 0.0
        if self._prev_gray is not None and self._prev_gray.shape == small.shape:
            motion = float(np.mean(cv2.absdiff(small, self._prev_gray))) / 255.0 * 6.0
        self._prev_gray = small
        motion = min(1.0, motion)
        sharpness = min(1.0, float(cv2.Laplacian(cv2.resize(gray, (320, 180)), cv2.CV_64F).var()) / 400.0)
        brightness = float(np.mean(gray)) / 255.0

        # associate faces with tracks
        assigned: set[str] = set()
        people: list[PersonObs] = []
        entrances: list[str] = []
        for face, (pid, sim) in zip(faces, matches):
            box = face.box
            best, best_iou = None, 0.0
            for tid, tr in self.tracks.items():
                if tid in assigned:
                    continue
                v = _iou(box, tr.box)
                if v > best_iou:
                    best, best_iou = tid, v
            if best is None or best_iou < 0.25:
                tid = f"{self.cam_id}-t{self._next}"
                self._next += 1
                tr = self.tracks[tid] = _Track(tid, box, now, now)
                entrances.append(tid)
            else:
                tid = best
                tr = self.tracks[tid]
            assigned.add(tid)
            tr.misses = 0
            tr.last_seen = now
            # local motion in the face box
            x, y, bw, bh = box
            x2, y2 = min(w, x + bw), min(h, y + bh)
            tr.motion = 0.9 * tr.motion + 0.1 * motion
            # mouth region from the landmarks: row[11],row[12] right mouth corner, row[13],row[14] left mouth corner
            row = face.row
            try:
                mx1, my1, mx2, my2 = float(row[10]), float(row[11]), float(row[12]), float(row[13])
                cx, cy = (mx1 + mx2) / 2, (my1 + my2) / 2
                mw = max(8, abs(mx2 - mx1) * 1.6)
                mh = max(6, mw * 0.6)
                rx1, ry1 = int(max(0, cx - mw / 2)), int(max(0, cy - mh / 2))
                rx2, ry2 = int(min(w, cx + mw / 2)), int(min(h, cy + mh / 2))
                mouth = cv2.resize(gray[ry1:ry2, rx1:rx2], (24, 14), interpolation=cv2.INTER_AREA) if ry2 > ry1 and rx2 > rx1 else None
                if mouth is not None and tr.mouth_prev is not None:
                    d = float(np.mean(cv2.absdiff(mouth, tr.mouth_prev))) / 255.0
                    # mouth motion relative to the face's own motion: talking moves the mouth more than the head
                    v = max(0.0, d * 8.0 - tr.motion * 0.5)
                    tr.speaking = 0.7 * tr.speaking + 0.3 * min(1.0, v)
                tr.mouth_prev = mouth
                # yaw: nose x relative to the eye midpoint, normalised by inter-eye distance
                ex1, ey1, ex2, ey2, nx = float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])
                eye_mid = (ex1 + ex2) / 2
                eye_d = max(1.0, abs(ex2 - ex1))
                tr.yaw = max(-1.0, min(1.0, 0.5 * tr.yaw + 0.5 * ((nx - eye_mid) / eye_d) * 2.0))
            except Exception:
                pass
            tr.box = box
            if pid:
                tr.person_id, tr.id_conf = pid, sim
            elif tr.person_id and sim > 0:
                tr.id_conf *= 0.9
            people.append(PersonObs(tid, tr.person_id if tr.id_conf >= 0.3 else None, tr.id_conf, box, bw / w,
                                    tr.speaking, tr.yaw, tr.motion, tr.first_seen, now))
        exits = []
        for tid in list(self.tracks):
            if tid not in assigned:
                self.tracks[tid].misses += 1
                if self.tracks[tid].misses > 12:  # ~2 s at 7 Hz: a real exit, not an occlusion blink
                    exits.append(tid)
                    del self.tracks[tid]
        # shot type from the biggest face
        if frozen or brightness < 0.02:   # only a truly black frame (lens cap, dead sensor) is unusable
            shot = "UNUSABLE"
        elif not people:
            shot = "EMPTY" if motion < 0.02 else "WIDE"
        else:
            fw = max(p.face_w_ratio for p in people)
            shot = "CLOSE_UP" if fw >= 0.28 else "MEDIUM_CLOSE_UP" if fw >= 0.16 else "MEDIUM" if fw >= 0.09 else "WIDE"
            if len(people) >= 2 and fw < 0.2:
                shot = "GROUP"
        return CameraObservation(self.cam_id, now, shot, people, motion, frozen, sharpness, brightness, entrances, exits, (w, h))
