"""Serve server/static over HTTPS on the LAN, so the sign-up camera works on a phone.

Copied from hemadassani/cue-desk-ui tools/serve_https.py (40fdd1a, PR #5); the only
change is that --root defaults to this repo's server/static, where the desk, dashboard
and sign-up pages live. The director on :8000 does not expose /static through the
tunnel, so this is the way a guest's phone on the same Wi-Fi reaches signup.html.

    python tools/serve_https.py                 # generates a cert, serves server/static on 8443
    python tools/serve_https.py --port 9000

Why this exists: phone browsers refuse getUserMedia unless the page is a secure
context, and http://192.168.x.x is not one. Over plain http the sign-up page can
only offer its file-upload fallback. A self-signed cert IS a secure context once
the browser has been told to trust it, so the two-photo camera flow works after
you accept the warning once.

Two mistakes worth not repeating, both of which cost time:

  * Bind to 0.0.0.0, not 127.0.0.1. Bound to loopback the page is fine on the
    laptop and completely unreachable from the phone, which Safari reports as
    "the server stopped responding" after a long wait -- it looks like a slow
    page rather than a wrong address.

  * The cert must carry the machine's LAN IP in subjectAltName, not just CN, or
    browsers reject it outright. --host handles that; pass the address the phone
    will actually use.
"""

import argparse
import http.server
import pathlib
import socket
import ssl
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def lan_ip() -> str:
    """The address a phone on the same network would use."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))       # no packets sent; just picks the route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(cert: pathlib.Path, key: pathlib.Path, host: str) -> None:
    if cert.exists() and key.exists():
        return
    cert.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "14",
         "-keyout", str(key), "-out", str(cert),
         "-subj", "/CN=" + host,
         "-addext", "subjectAltName=IP:%s,IP:127.0.0.1,DNS:localhost" % host],
        check=True,
        # Git Bash rewrites a leading /CN= into a Windows path unless this is set
        env={**__import__("os").environ, "MSYS_NO_PATHCONV": "1"},
    )
    print("generated a 14-day self-signed cert for %s" % host, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(REPO / "server" / "static"))
ap.add_argument("--port", type=int, default=8443)
ap.add_argument("--host", default=None, help="the LAN IP to put in the cert (default: detected)")
ap.add_argument("--cert", default=None)
ap.add_argument("--key", default=None)
args = ap.parse_args()

ROOT = pathlib.Path(args.root)
PORT = args.port
HOST = args.host or lan_ip()
CERT = pathlib.Path(args.cert) if args.cert else REPO / ".certs" / "dev-cert.pem"
KEY = pathlib.Path(args.key) if args.key else REPO / ".certs" / "dev-key.pem"
ensure_cert(CERT, KEY, HOST)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        # never let a phone cache a stale build while we are iterating
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stdout.write("%s %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()


ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))

# 0.0.0.0, deliberately: bound to loopback the phone cannot reach this at all.
srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
print("serving %s" % ROOT, flush=True)
print("  this machine : https://127.0.0.1:%d/dashboard.html" % PORT, flush=True)
print("  a phone      : https://%s:%d/signup.html" % (HOST, PORT), flush=True)
print("  the phone will warn about the certificate. Accept it once.", flush=True)
srv.serve_forever()
