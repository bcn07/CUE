# iPhone cameras for CUE

For: D (director and Mac runtime), with A for the camera page
From: C
Goal: replace the three laptop webcams with iPhones, add zoom and pan, spend nothing, and keep it secure.

## The one real blocker

iPhone Safari only gives camera access to HTTPS pages. The director runs on plain `http://10.x.x.x:8000`. The Chrome flag trick in the laptop README does not exist on iPhones. So the whole job is this: give the director an HTTPS address.

## Step 1: the 15 minute test (do this before anything else)

On the Mac:

```
brew install cloudflared
cloudflared tunnel --url http://localhost:8000
```

It prints a random `https://<words>.trycloudflare.com` address. No account, no payment. WebSockets work through it.

On one iPhone, open Safari and go to:

```
https://<words>.trycloudflare.com/cam?cam=B
```

Allow the camera. If the picture shows up on the director, iPhones are viable. If it does not work in 15 minutes, we stay on laptops for the demo.

Why a tunnel and not local certificates: no certificate installs on three phones, and it still works if the venue Wi-Fi blocks devices from talking to each other. Phones can even use mobile data.

## Step 2: keep it secure

The tunnel address is public, so lock it down:

- Join code. `/cam` and `/ingest` require `?code=<4 to 6 digits>`. The code shows on `/setup`, is per event, and can be rotated. Wrong code gets rejected.
- Only `/cam` and `/ingest` are reachable through the tunnel. When the Host header is a `trycloudflare.com` name, everything else returns 404. `/setup` and the producer desk stay local.
- One publisher per camera id. A second device can replace the first only after the director approves.
- Video only from the phones. No microphone, except the one phone we choose as the master mic.
- Close the tunnel right after the demo.
- Never put API keys in any page the phones load.

## Step 3: zoom and pan for free (virtual PTZ)

No motors, no apps, no cost.

- The phone captures at full quality, 1080p or 4K.
- The page draws a cropped window of that frame onto a 1280x720 canvas and sends that as the JPEG.
- Moving the window is a pan. Shrinking it is a zoom.
- The director controls it remotely over the same socket:

```
{"type": "ptz", "cam": "B", "x": 0.5, "y": 0.4, "zoom": 1.8}
```

- The page eases to the target over about 400 ms so it looks like a camera move, not a jump.
- Zoom range 1.0 to 3.0, clamped so the window never leaves the frame. With 4K input, 3x still looks sharp at 720p.
- Safari exposes the ultra wide and telephoto lenses on recent iPhones. Ultra wide is perfect for the safe wide shot.

Real physical panning is not possible without hardware. This is the standard software answer.

## Phone setup so nothing dies mid-demo

- Plugged in. Low Power Mode off.
- Settings, Display, Auto-Lock: Never.
- Do Not Disturb on, so a call cannot kill the stream.
- Guided Access (triple click the side button) locks the phone into Safari.
- Landscape, on a tripod or a stack of books. Do not move it after framing is approved.
- Keep Safari in the foreground. iOS pauses the camera in the background.

## Prompt for your coding agent

```
Add iPhone camera support to the director, free and secure.
1. The /cam page must work in iOS Safari over HTTPS: playsinline, muted,
   a Start button (user gesture), rear camera by default, a lens picker
   from enumerateDevices (wide, ultra wide, telephoto when exposed),
   request 1920x1080 at 30 fps, Wake Lock where supported, clear
   LIVE / STANDBY / OFFLINE text, auto-reconnect.
2. Join code: /cam and /ingest require ?code=<4-6 digits> shown on the
   /setup page, per event, rotatable. Reject wrong codes. One publisher
   per camera id; a new one replaces the old only after the director
   approves.
3. Virtual PTZ: the page draws a crop window of the full-resolution frame
   onto a 1280x720 canvas and sends that as the JPEG. The director sends
   {type:"ptz", cam, x, y, zoom} over the same socket; the page eases to
   the target over 400 ms. Zoom 1.0 to 3.0, clamped so the window never
   leaves the frame. In the producer UI add drag-to-pan, wheel-to-zoom,
   a Reset button and three presets per camera.
4. Document `cloudflared tunnel --url http://localhost:8000`. When the
   Host header is a trycloudflare.com name, serve only /cam and /ingest
   and return 404 for everything else.
5. Test on a real iPhone: start, lock screen, unlock, incoming call,
   Wi-Fi drop. The stream must recover or show OFFLINE clearly.
```

## Risks and the fallback

| Risk | What we do |
|---|---|
| Tunnel adds delay or stutters on venue internet | Drop to 15 fps and JPEG quality 60. If still bad, go back to laptops. |
| Safari pauses the camera (lock, call, app switch) | Guided Access, Do Not Disturb, Auto-Lock Never, OFFLINE shown clearly on the director. |
| Phone overheats | Out of direct light, case off, lower resolution. |
| Tunnel address changes on restart | Keep the tunnel running. Re-share the link only if it restarts. |
| Too much change the night before | Time-box it. Laptops with `run_camera.bat` remain the fallback and already work. |

## Decision rule

1. Tunnel plus one iPhone on the existing `/cam` page works in 15 minutes: we go iPhones.
2. Join code and the 404 rule next, because security is not optional.
3. Virtual PTZ is a Sunday stretch, only after the basic three-camera demo is solid.

## One upside for the speech lane

A phone next to the host is a far better microphone than a laptop across the room. If the host phone's audio becomes the master mic, Deepgram accuracy goes up and CUE's decisions get more reliable. Still exactly one master mic, never more.
