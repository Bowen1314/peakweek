#!/usr/bin/env python3
"""peakweek web app (stdlib only).

    python3 server.py                                         # live: runs the TabPFN forecast
    python3 server.py --fixture examples/forecast_fixture.json  # no TabPFN: serves a saved forecast

Endpoints
    GET /                              the page (static/)
    GET /api/info                      {"mode": "live" | "fixture", ...}
    GET /api/geocode?q=                place search (or "lat, lon")
    GET /api/forecast?lat=&lon=&name=  start (or reuse) a forecast job -> job status
    GET /api/forecast/status?id=       job status; when done it carries the annotated result and card HTML

A forecast takes ~20-60 s on a 4-core CPU, so it runs on one background worker thread; requests
are queued and only one forecast is computed at a time (memory). Results are cached in memory per
(lat, lon rounded to 0.01 degree, date). TabPFN is never imported at module import time: the live
forecast function imports peakweek.forecast on first use.

Each new job also asks iNaturalist which of the trees are recorded within 50 km (peakweek.nearby):
of the requested place in live mode, of the fixture's own place in fixture mode (the card shows the
fixture's numbers, so the check must match them). That call runs on its own short-lived thread,
started when the job is submitted, so it never occupies the forecast worker; the worker waits for it
at most NEARBY_WAIT seconds after the forecast and otherwise marks the local check unavailable.

In fixture mode with a saved real (non-synthetic) forecast, each result carries a `replay` note
{place, made, searched} that the card shows as a banner ("Replaying a saved forecast for ...").
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import queue
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import parse_qs, urlsplit

from peakweek import card, geocode, nearby, verdict

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
REAL_EXAMPLE = ROOT / "examples" / "forecast_new_brunswick.json"
DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 8770
MAX_JOBS = 200
NEARBY_WAIT = 15.0  # seconds; longer than nearby.counts' own 10 s timeout
SAME_PLACE_DEG = 0.05  # a search this close to the fixture's place counts as the same place
MAX_NAME = 120

# Allowlist: request path -> (file in static/, content type). Nothing else is served from disk.
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def expect_from(result: Optional[dict]) -> Optional[dict]:
    """What a forecast run involves, read from a result's model/weather fields (never made up)."""
    if not result:
        return None
    model = result.get("model") or {}
    weather = result.get("weather") or {}
    e = {}
    if weather.get("past_days"):
        e["past_days"] = int(weather["past_days"])
    if weather.get("archive_from"):
        e["weather_since"] = str(weather["archive_from"])[:10]
    if model.get("context_rows"):
        e["context_rows"] = int(model["context_rows"])
    if model.get("seconds") is not None:
        e["seconds"] = float(model["seconds"])
    return e or None


def replay_info(data: dict, lat: float, lon: float, name: Optional[str]) -> dict:
    """The `replay` note for a saved forecast served in fixture mode, for a search at (lat, lon)."""
    place = data.get("place") or {}
    info = {"place": card.place_name(data), "made": str(data["today"])[:10] if data.get("today") else None,
            "searched": None}
    plat, plon = place.get("lat"), place.get("lon")
    if plat is None or plon is None or abs(lat - float(plat)) > SAME_PLACE_DEG or abs(lon - float(plon)) > SAME_PLACE_DEG:
        info["searched"] = name or "%.2f, %.2f" % (lat, lon)
    return info


def live_forecast(lat: float, lon: float, name: Optional[str]) -> dict:
    from peakweek.forecast import forecast  # lazy: pulls in TabPFN
    return forecast(lat, lon, place_name=name)


class Forecaster:
    """Queue of forecast jobs, one worker thread, in-memory cache of finished results."""

    def __init__(self, forecast_fn: Callable, mode: str, expect: Optional[dict] = None,
                 today_fn: Callable[[], _dt.date] = _dt.date.today, quiet: bool = False,
                 nearby_fn: Optional[Callable] = nearby.counts, nearby_wait: float = NEARBY_WAIT):
        self.forecast_fn = forecast_fn
        self.nearby_fn = nearby_fn
        self.nearby_wait = nearby_wait
        self.quiet = quiet
        self.mode = mode
        self.expect = expect
        self.today_fn = today_fn
        self._lock = threading.Lock()
        self._jobs = {}
        self._by_key = {}
        self._pending = []
        self._running = None
        self._q = queue.Queue()
        threading.Thread(target=self._loop, name="peakweek-forecast", daemon=True).start()

    def submit(self, lat: float, lon: float, name: Optional[str]) -> dict:
        key = (round(lat, 2), round(lon, 2), self.today_fn().isoformat())
        with self._lock:
            jid = self._by_key.get(key)
            if jid in self._jobs and self._jobs[jid]["status"] != "error":
                snap = self._snapshot(jid)
                snap["cached"] = True
                return snap
            jid = uuid.uuid4().hex[:12]
            job = {"id": jid, "key": key, "lat": lat, "lon": lon, "name": name,
                   "status": "queued", "submitted": time.time(),
                   "nearby": None, "nearby_done": threading.Event()}
            self._jobs[jid] = job
            self._by_key[key] = jid
            self._pending.append(jid)
            self._trim()
            snap = self._snapshot(jid)
        if self.nearby_fn is None:
            job["nearby_done"].set()
        else:
            threading.Thread(target=self._lookup_nearby, args=(job,), name="peakweek-nearby", daemon=True).start()
        self._q.put(jid)
        return snap

    def _lookup_nearby(self, job: dict) -> None:
        """Runs on its own thread (never the forecast worker): the iNaturalist local check."""
        try:
            job["nearby"] = self.nearby_fn(job["lat"], job["lon"])
        except Exception:  # the check is optional; any failure means "unavailable"
            if not self.quiet:
                traceback.print_exc()
            job["nearby"] = None
        finally:
            job["nearby_done"].set()

    def status(self, jid: str) -> Optional[dict]:
        with self._lock:
            return self._snapshot(jid) if jid in self._jobs else None

    def _snapshot(self, jid: str) -> dict:
        job = self._jobs[jid]
        now = time.time()
        snap = {"id": jid, "status": job["status"], "mode": self.mode, "expect": self.expect,
                "place": {"lat": job["lat"], "lon": job["lon"], "name": job["name"]},
                "elapsed": round(now - job["submitted"], 1)}
        if job["status"] == "queued":
            snap["ahead"] = (self._pending.index(jid) if jid in self._pending else 0) + (1 if self._running else 0)
        elif job["status"] == "done":
            snap["result"] = job["result"]
            snap["card_html"] = job["card_html"]
            snap["seconds"] = round(job["finished"] - job["started"], 1)
        elif job["status"] == "error":
            snap["error"] = job["error"]
        return snap

    def _trim(self) -> None:
        finished = [j for j in self._jobs.values() if j["status"] in ("done", "error")]
        finished.sort(key=lambda j: j["submitted"])
        while len(self._jobs) > MAX_JOBS and finished:
            old = finished.pop(0)
            del self._jobs[old["id"]]
            if self._by_key.get(old["key"]) == old["id"]:
                del self._by_key[old["key"]]

    def _loop(self) -> None:
        while True:
            jid = self._q.get()
            with self._lock:
                job = self._jobs.get(jid)
                if jid in self._pending:
                    self._pending.remove(jid)
                if job is None:
                    continue
                job["status"] = "running"
                job["started"] = time.time()
                self._running = jid
            try:
                raw = self.forecast_fn(job["lat"], job["lon"], job["name"])
                local = job["nearby"] if job["nearby_done"].wait(self.nearby_wait) else None
                annotated = verdict.annotate(raw, local)
                html = card.render_html(annotated)
            except Exception as e:  # report any failure to the page instead of dying
                if not self.quiet:
                    traceback.print_exc()
                with self._lock:
                    job["status"] = "error"
                    job["error"] = "%s: %s" % (type(e).__name__, e)
                    self._running = None
                    if self._by_key.get(job["key"]) == jid:
                        del self._by_key[job["key"]]  # let the next request retry
                continue
            with self._lock:
                job.update(status="done", result=annotated, card_html=html, finished=time.time())
                self._running = None
                if self.mode == "live":
                    self.expect = expect_from(annotated) or self.expect


class BadRequest(Exception):
    pass


def _one(q: dict, name: str) -> str:
    vals = q.get(name)
    return vals[0] if vals else ""


def _coord(q: dict, name: str, lo: float, hi: float) -> float:
    raw = _one(q, name).strip()
    try:
        v = float(raw)
    except ValueError:
        raise BadRequest("%s must be a number" % name)
    if not math.isfinite(v) or not (lo <= v <= hi):
        raise BadRequest("%s must be between %g and %g" % (name, lo, hi))
    return v


class Handler(BaseHTTPRequestHandler):
    server_version = "peakweek/0.1"

    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        q = parse_qs(parts.query)
        try:
            if parts.path in STATIC_FILES:
                return self._static(parts.path)
            if parts.path == "/api/info":
                return self._json(200, self.server.app_info)
            if parts.path == "/api/geocode":
                return self._geocode(q)
            if parts.path == "/api/forecast":
                return self._forecast(q)
            if parts.path == "/api/forecast/status":
                return self._status(q)
            self._json(404, {"error": "not found"})
        except BadRequest as e:
            self._json(400, {"error": str(e)})
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": "internal error"})

    def _static(self, path: str) -> None:
        fname, ctype = STATIC_FILES[path]
        body = (self.server.static_dir / fname).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _geocode(self, q: dict) -> None:
        text = _one(q, "q").strip()[:MAX_NAME]
        if not text:
            raise BadRequest("type a place name")
        try:
            ll = geocode.parse_latlon(text)
        except ValueError as e:
            raise BadRequest(str(e))
        if ll is not None:
            return self._json(200, {"results": [geocode.latlon_place(*ll)]})
        try:
            results = self.server.geocode_fn(text)
        except Exception as e:
            return self._json(502, {"error": "Place search failed (%s). You can also type coordinates like "
                                             "40.49, -74.45." % type(e).__name__})
        self._json(200, {"results": results})

    def _forecast(self, q: dict) -> None:
        lat = _coord(q, "lat", -90.0, 90.0)
        lon = _coord(q, "lon", -180.0, 180.0)
        name = " ".join(_one(q, "name").split())[:MAX_NAME] or None
        self._json(200, self.server.forecaster.submit(lat, lon, name))

    def _status(self, q: dict) -> None:
        snap = self.server.forecaster.status(_one(q, "id"))
        if snap is None:
            return self._json(404, {"error": "unknown forecast id (the server may have restarted)"})
        self._json(200, snap)

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args) -> None:
        if self.server.quiet or "/api/forecast/status" in self.path:
            return  # the page polls once a second; keep the log readable
        super().log_message(fmt, *args)


def _load_json(source: Union[str, Path, dict]) -> dict:
    if isinstance(source, dict):
        return source
    with open(source, encoding="utf-8") as f:
        return json.load(f)


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *,
                fixture: Union[str, Path, dict, None] = None, fixture_delay: float = 0.0,
                forecast_fn: Optional[Callable] = None, geocode_fn: Optional[Callable] = None,
                nearby_fn: Optional[Callable] = nearby.counts,
                seed_example: Optional[Path] = REAL_EXAMPLE, static_dir: Path = STATIC_DIR,
                quiet: bool = False) -> ThreadingHTTPServer:
    """Build the server (port 0 picks a free port). Fakes can be injected for tests.

    `nearby_fn(lat, lon)` is called with the requested place in live mode and with the fixture's own
    place in fixture mode (falling back to the request if the fixture has no coordinates).
    `nearby_fn=None` skips the check (the card then says it was unavailable).
    """
    info = {"mode": "live"}
    expect = None
    if fixture is not None:
        data = _load_json(fixture)
        verdict.annotate(data)  # fail fast on a malformed fixture

        def fixture_forecast(lat, lon, name):
            if fixture_delay:
                time.sleep(fixture_delay)
            r = copy.deepcopy(data)
            if not data.get("synthetic"):
                r["replay"] = replay_info(data, lat, lon, name)
            return r

        forecast_fn = forecast_fn or fixture_forecast
        place = data.get("place") or {}
        if nearby_fn is not None and place.get("lat") is not None and place.get("lon") is not None:
            fixture_ll = (float(place["lat"]), float(place["lon"]))
            check_place = nearby_fn

            def fixture_nearby(lat, lon):  # the card shows the fixture's numbers: check the fixture's place
                return check_place(*fixture_ll)

            nearby_fn = fixture_nearby
        expect = expect_from(data)
        info = {"mode": "fixture", "fixture_place": (data.get("place") or {}).get("name"),
                "synthetic": bool(data.get("synthetic"))}
    else:
        forecast_fn = forecast_fn or live_forecast
        if seed_example is not None and Path(seed_example).exists():
            try:
                expect = expect_from(_load_json(seed_example))
            except (OSError, ValueError):
                expect = None
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.forecaster = Forecaster(forecast_fn, info["mode"], expect, quiet=quiet, nearby_fn=nearby_fn)
    server.geocode_fn = geocode_fn or geocode.search
    server.app_info = info
    server.static_dir = Path(static_dir)
    server.quiet = quiet
    return server


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="peakweek web app")
    p.add_argument("--host", default=DEFAULT_HOST, help="default 127.0.0.1 (this machine only)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--fixture", metavar="PATH", help="serve this saved forecast() result instead of running TabPFN")
    p.add_argument("--fixture-delay", type=float, default=0.0, metavar="SECONDS",
                   help="pretend a fixture forecast takes this long (to see the waiting state)")
    args = p.parse_args(argv)
    server = make_server(args.host, args.port, fixture=args.fixture, fixture_delay=args.fixture_delay)
    host, port = server.server_address[:2]
    mode = ("fixture mode: every search shows %s" % args.fixture) if args.fixture else "live mode (TabPFN)"
    print("peakweek on http://%s:%d/  (%s)" % (host, port, mode), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
