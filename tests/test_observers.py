import time

import cv2
import numpy as np

from server.identity import Face
from server.observers import CameraObserver


def face_at(x, y, w, h, mouth_dy=0):
    # YuNet row: x,y,w,h, right eye, left eye, nose, right mouth, left mouth, score
    row = np.array([x, y, w, h, x + w * 0.3, y + h * 0.35, x + w * 0.7, y + h * 0.35, x + w * 0.5, y + h * 0.55,
                    x + w * 0.35, y + h * 0.75 + mouth_dy, x + w * 0.65, y + h * 0.75 + mouth_dy, 0.95], dtype=np.float32)
    return Face((x, y, w, h), 0.95, row)


def frame(mouth_open: bool, noise=0):
    img = np.full((360, 640, 3), 120, np.uint8)
    cv2.rectangle(img, (250, 100), (390, 280), (200, 180, 160), -1)              # a "face"
    cv2.rectangle(img, (295, 230), (345, 250 if not mouth_open else 262), (40, 20, 20), -1)   # the mouth
    if noise:
        img = cv2.add(img, np.random.randint(0, noise, img.shape, dtype=np.uint8))
    return img


def test_mouth_motion_raises_speaking_and_stillness_lowers_it():
    ob = CameraObserver("C")
    f = face_at(250, 100, 140, 180)
    t = 0.0
    for i in range(16):
        o = ob.observe(frame(mouth_open=(i % 2 == 0)), [f], [("sarah", 0.8)], t)
        t += 0.15
    talking = o.people[0].speaking
    assert o.people[0].person_id == "sarah" and o.shot_type in ("MEDIUM_CLOSE_UP", "MEDIUM", "CLOSE_UP") and not o.frozen
    for i in range(16):
        o = ob.observe(frame(mouth_open=False), [f], [("sarah", 0.8)], t)
        t += 0.15
    quiet = o.people[0].speaking
    assert talking > 0.3 and quiet < talking * 0.5, (talking, quiet)
    assert o.frozen and o.shot_type == "UNUSABLE"  # identical frames in a row: a still image or a stuck webcam


def test_tracks_entrances_exits_and_shot_types():
    ob = CameraObserver("A")
    o = ob.observe(frame(False, noise=8), [], [], 0.0)
    assert o.shot_type in ("EMPTY", "WIDE") and not o.people
    f = face_at(250, 100, 30, 40)   # a small face: wide framing
    o = ob.observe(frame(False, noise=8), [f], [(None, 0.0)], 0.2)
    assert o.entrances and o.shot_type == "WIDE" and o.people[0].person_id is None
    t = 0.4
    exits = []
    for _ in range(14):               # the face leaves; after ~2 s of misses it is an exit
        o = ob.observe(frame(False, noise=8), [], [], t)
        exits += o.exits
        t += 0.15
    assert exits
