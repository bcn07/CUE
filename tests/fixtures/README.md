# Test fixtures

The tests enrol and match faces using three photos of public figures, so they
never need photos of real attendees.

| File | Source | Original photo (from the embedded EXIF) |
|---|---|---|
| `obama.jpg` | `examples/obama.jpg` in [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) | Official White House photo by Pete Souza, Oval Office portrait sitting, 6 Dec 2012 |
| `obama2.jpg` | `examples/obama2.jpg` in the same repo | Official White House photo by Pete Souza, address to a joint session of Congress, 9 Sep 2009 |
| `biden.jpg` | `examples/biden.jpg` in the same repo | Official White House photo by Pete Souza, Blue Room, 10 May 2010 (cropped) |

All three files are byte-identical to the copies in `ageitgey/face_recognition`
(same git blob hashes). That repository is MIT licensed.

Photos taken by US federal employees as part of their official duties are not
subject to copyright in the United States (17 U.S.C. § 105). The White House
usage note in the EXIF also asks that the photos not be manipulated or used in a
way that suggests endorsement by the President, the First Family or the White
House. Here they are used only as test inputs for face detection and matching.
