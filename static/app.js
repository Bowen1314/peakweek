// peakweek page: search -> forecast job (polled) -> card. The card HTML is rendered (and escaped)
// by the server's peakweek/card.py; this script only moves between views and fills in plain text.
//
// Two modes. Live (python3 server.py): the server's JSON API under /api/. Static (the daily GitHub Pages
// site, built by scripts/build_site.py; its index.html carries <meta name="peakweek-static">): there is
// no server, so apiGet() answers the same requests from the precomputed files under ./data/ (relative
// paths: the site lives at /peakweek/), and place search calls Open-Meteo's geocoding API directly.
"use strict";

// BEGIN static helpers (pure, no DOM; tests/test_static_mode.py runs this block under node and compares
// it with peakweek/geocode.py, peakweek/config.py and peakweek/weather.py)
const PeakweekStatic = (function () {
  // Same table as peakweek/geocode.py US_STATES.
  const US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
  };
  const NUM = "[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)";
  const LATLON = new RegExp("^\\s*(" + NUM + ")\\s*(?:,|\\s)\\s*(" + NUM + ")\\s*$");
  const GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search";

  // geocode.parse_latlon: [lat, lon], null if not a coordinate pair; throws if out of range.
  function parseLatLon(text) {
    const m = LATLON.exec(text || "");
    if (!m) return null;
    const lat = parseFloat(m[1]), lon = parseFloat(m[2]);
    if (!(lat >= -90 && lat <= 90)) throw new Error("latitude must be between -90 and 90");
    if (!(lon >= -180 && lon <= 180)) throw new Error("longitude must be between -180 and 180");
    return [lat, lon];
  }

  // geocode.latlon_place
  function latlonPlace(lat, lon) {
    const label = lat.toFixed(4) + ", " + lon.toFixed(4);
    return { name: label, label: label, short: label, lat: lat, lon: lon,
             admin1: null, country: null, country_code: null };
  }

  // geocode.parse_response
  function parsePlaces(data) {
    const out = [];
    for (const r of (data && data.results) || []) {
      const lat = Number(r.latitude), lon = Number(r.longitude);
      if (r.latitude == null || r.longitude == null || !Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      const name = String(r.name || "").trim();
      if (!name) continue;
      const admin1 = r.admin1 ?? null, country = r.country ?? null, cc = r.country_code ?? null;
      const label = [name, admin1, country].filter(Boolean).join(", ");
      const short = (cc === "US" && Object.prototype.hasOwnProperty.call(US_STATES, admin1))
        ? name + ", " + US_STATES[admin1] : [name, admin1 || country].filter(Boolean).join(", ");
      out.push({ name: name, label: label, short: short, lat: lat, lon: lon, admin1: admin1, country: country,
                 country_code: cc, elevation_m: r.elevation ?? null });
    }
    return out;
  }

  function geocodeURL(name) {
    return GEOCODE_URL + "?" + new URLSearchParams({ name: name, count: "5", language: "en", format: "json" });
  }

  // config.in_region (inclusive bounds)
  function inRegion(region, lat, lon) {
    return region.lat_min <= lat && lat <= region.lat_max && region.lon_min <= lon && lon <= region.lon_max;
  }

  // The published cell containing the point: bounds come from data/index.json and are half-open,
  // which is the cell weather.cell_id() assigns (floor of lat / GRID_DEG). No cell_id arithmetic here.
  function findCell(cells, lat, lon) {
    for (const c of cells) {
      if (c.lat_min <= lat && lat < c.lat_max && c.lon_min <= lon && lon < c.lon_max) return c;
    }
    return null;
  }

  return { US_STATES, parseLatLon, latlonPlace, parsePlaces, geocodeURL, inRegion, findCell };
})();
// END static helpers

(function () {
  const $ = (id) => document.getElementById(id);
  const VIEWS = ["search", "waiting", "error", "result", "outside"];
  let info = { mode: "live" };
  let current = null;      // {lat, lon, name} being forecast
  let lastJob = null;      // latest job status from the server
  let startedAt = 0;
  let pollTimer = null;
  let tickTimer = null;

  function show(view) {
    for (const v of VIEWS) $(v).hidden = v !== view;
    document.body.dataset.view = view;
  }

  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    let body = {};
    try { body = await r.json(); } catch (_) { /* not JSON */ }
    if (!r.ok) throw new Error(body.error || ("Request failed (HTTP " + r.status + ")"));
    return body;
  }

  // ---- static mode: the API answered from the daily files (see the header comment)
  const S = PeakweekStatic;
  const staticMeta = document.querySelector('meta[name="peakweek-static"]');
  const STATIC_INDEX = staticMeta ? "./" + staticMeta.getAttribute("content") : null;  // "./data/index.json"
  if (STATIC_INDEX) info = { mode: "static" };
  let indexPromise = null;

  function loadIndex() {
    if (!indexPromise) indexPromise = getJSON(STATIC_INDEX).catch((e) => { indexPromise = null; throw e; });
    return indexPromise;
  }

  function apiGet(url) {
    return STATIC_INDEX ? staticGet(url) : getJSON(url);
  }

  async function staticGet(url) {
    const [path, query] = url.split("?");
    const q = new URLSearchParams(query || "");
    if (path === "/api/info") return Object.assign({}, await loadIndex(), { mode: "static" });
    if (path === "/api/geocode") return staticGeocode(q.get("q") || "");
    if (path === "/api/forecast") return staticForecast(Number(q.get("lat")), Number(q.get("lon")), q.get("name") || "");
    throw new Error("Not available on the static site.");
  }

  async function staticGeocode(text) {
    text = text.trim().slice(0, 120);
    if (!text) throw new Error("type a place name");
    const ll = S.parseLatLon(text);
    if (ll) return { results: [S.latlonPlace(ll[0], ll[1])] };
    try {
      let results = S.parsePlaces(await getJSON(S.geocodeURL(text)));
      if (!results.length && text.includes(",")) {  // "Town, ST" did not match; try the town alone
        results = S.parsePlaces(await getJSON(S.geocodeURL(text.split(",")[0].trim())));
      }
      return { results };
    } catch (e) {
      throw new Error("Place search failed (" + (e.name || "error") + "). You can also type coordinates like 40.49, -74.45.");
    }
  }

  async function staticForecast(lat, lon, name) {
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
      throw new Error("That is not a valid place (latitude -90 to 90, longitude -180 to 180).");
    }
    const index = await loadIndex();
    const where = name || (lat.toFixed(2) + ", " + lon.toFixed(2));
    if (!S.inRegion(index.region, lat, lon)) {
      throw new Error(where + " is outside the area this site forecasts. " + index.region_note +
        " The daily forecasts here cover only that area; try a place inside it.");
    }
    const cell = S.findCell(index.cells, lat, lon);
    if (!cell) {
      throw new Error(where + " is inside the area, but its 1\u00b0 grid cell has no forecast: the training data has " +
        "no observations there (mostly open water or remote land). Try a town nearby.");
    }
    const job = await getJSON("./data/" + cell.file);
    job.place = { lat: lat, lon: lon, name: where };
    job.result.place.name = where;  // the card and the "go outside" view show the name the user searched
    return job;
  }

  // New York date of the browser's clock, "YYYY-MM-DD" (the daily build dates forecasts the same way).
  function newYorkToday() {
    try {
      return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(new Date());
    } catch (_) {
      return null;
    }
  }

  const fmtInt = (n) => Number(n).toLocaleString("en-US");
  const round4 = (x) => Math.round(x * 1e4) / 1e4;
  function setMsg(text) { $("search-msg").textContent = text || ""; }

  async function loadInfo() {
    try { info = await apiGet("/api/info"); } catch (_) { return; }
    if (info.mode === "static") {
      let text = "Forecasts made " + info.today_label + " (New York date). To run without a server, the model " +
        "forecasts once every morning for each 1\u00b0 grid cell (about 110 by 80 km) of the area; your card is " +
        "for the cell around your place, not your exact spot. Updated daily.";
      const now = newYorkToday();
      if (now && info.today && (Date.parse(now) - Date.parse(info.today)) / 864e5 >= 2) {
        text += " The daily update has not run since " + info.today_label + ", so this forecast is out of date.";
      }
      if (info.synthetic) text = "Synthetic test build: a fake model and made-up weather, not a forecast. " + text;
      $("mode-note").textContent = text;
      $("mode-note").hidden = false;
    }
    if (info.mode === "fixture") {
      const what = info.synthetic ? "synthetic example forecast" : "saved forecast";
      $("mode-note").textContent = "Demo server: every search shows the " + what + " for " +
        (info.fixture_place || "one place") + ". Run the server without --fixture for live forecasts.";
      $("mode-note").hidden = false;
    }
  }

  // ---- search
  $("search-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const q = $("q").value.trim();
    if (!q) return;
    $("places").hidden = true;
    setMsg("Searching…");
    try {
      const data = await apiGet("/api/geocode?q=" + encodeURIComponent(q));
      const results = data.results || [];
      if (!results.length) {
        setMsg("No place found. Try “Town, ST” or coordinates like 40.49, -74.45.");
        return;
      }
      if (results.length === 1 && !results[0].country_code) {  // typed coordinates
        setMsg("");
        start({ lat: results[0].lat, lon: results[0].lon, name: results[0].short });
        return;
      }
      setMsg(results.length === 1 ? "Is this the place?" : "Which one?");
      renderPlaces(results);
    } catch (e) {
      setMsg(e.message);
    }
  });

  function renderPlaces(results) {
    const ul = $("places");
    ul.textContent = "";
    for (const p of results) {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.className = "place-btn";
      b.textContent = p.label;
      b.addEventListener("click", () => start({ lat: p.lat, lon: p.lon, name: p.short || p.name }));
      li.appendChild(b);
      ul.appendChild(li);
    }
    ul.hidden = false;
  }

  $("locate").addEventListener("click", () => {
    if (!navigator.geolocation) {
      setMsg("This browser cannot share a location. Type a place instead.");
      return;
    }
    setMsg("Asking the browser for your location…");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setMsg("");
        const lat = round4(pos.coords.latitude), lon = round4(pos.coords.longitude);
        start({ lat, lon, name: lat.toFixed(2) + ", " + lon.toFixed(2) + " (your location)" });
      },
      (err) => setMsg("Location not available (" + err.message + "). Type a place instead."),
      { enableHighAccuracy: false, timeout: 15000, maximumAge: 600000 }
    );
  });

  // ---- forecast job
  function stopTimers() {
    clearTimeout(pollTimer);
    clearInterval(tickTimer);
    pollTimer = tickTimer = null;
  }

  async function start(place) {
    stopTimers();
    current = place;
    const params = new URLSearchParams({ lat: place.lat, lon: place.lon, name: place.name || "" });
    history.replaceState(null, "", "?" + params.toString());
    lastJob = null;
    startedAt = Date.now();
    renderWaiting();
    show("waiting");
    tickTimer = setInterval(renderWaiting, 1000);
    try {
      handle(await apiGet("/api/forecast?" + params.toString()));
    } catch (e) {
      fail(e.message);
    }
  }

  function handle(job) {
    lastJob = job;
    if (job.status === "done") return finish(job);
    if (job.status === "error") return fail("The forecast failed: " + job.error);
    renderWaiting();
    pollTimer = setTimeout(async () => {
      try {
        handle(await apiGet("/api/forecast/status?id=" + encodeURIComponent(job.id)));
      } catch (e) {
        fail(e.message);
      }
    }, 1000);
  }

  // Honest waiting text: numbers come from the server's `expect` (a real result's model/weather
  // fields). With none known, the text has no numbers.
  function renderWaiting() {
    const job = lastJob;
    const e = (job && job.expect) || {};
    const mode = (job && job.mode) || info.mode;
    const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    const since = e.weather_since ? MONTHS[Number(e.weather_since.slice(5, 7)) - 1] + " " + Number(e.weather_since.slice(8, 10)) : "";
    const weather = since ? "weather since " + since + " and a 2-week forecast"
      : (e.past_days ? fmtInt(e.past_days) + " days of weather" : "recent weather");
    const rows = e.context_rows ? fmtInt(e.context_rows) + " past observations" : "past iNaturalist observations";
    const steps = "fetching " + weather + " from Open-Meteo, then running TabPFN on " + rows;
    const secs = Math.max(0, Math.round((Date.now() - startedAt) / 1000));
    $("wait-place").textContent = "Forecasting for " + ((current && current.name) || "your place");
    let what;
    if (mode === "static") {
      what = "Loading the forecast made " + (info.today_label || "this morning") + " for this area…";
    } else if (mode === "fixture") {
      what = "Loading the saved example. (A live forecast means " + steps + ".)";
    } else if (job && job.status === "queued" && job.ahead > 0) {
      what = "Waiting for " + (job.ahead === 1 ? "another forecast" : job.ahead + " other forecasts") +
        " to finish first (one at a time). Then: " + steps + "…";
    } else {
      what = steps.charAt(0).toUpperCase() + steps.slice(1) + "…";
    }
    $("wait-what").textContent = what;
    let time = secs + " s so far";
    if (e.seconds && mode === "live") time += " · a recent run took " + Math.round(e.seconds) + " s";
    $("wait-time").textContent = time;
  }

  function finish(job) {
    stopTimers();
    $("card").innerHTML = job.card_html;  // rendered and escaped server-side (peakweek/card.py)
    const r = job.result || {};
    if (job.cell) {  // static mode: the card is the grid cell's; say so, under the name the user searched
      const placeEl = document.querySelector("#card .card-head .place");
      if (placeEl) placeEl.textContent = (r.place && r.place.name) || "";
      const madeEl = document.querySelector("#card .card-head .made");
      if (madeEl) madeEl.textContent = job.cell.note;
    }
    const place = (r.place && r.place.name) || (current && current.name) || "";
    const wk = document.querySelector("#card .weekend strong");
    $("out-when").textContent = [place, wk ? "weekend of " + wk.textContent : ""].filter(Boolean).join(" · ");
    $("out-headline").textContent = r.headline || "";
    document.title = "peakweek: " + place;
    show("result");
    window.scrollTo(0, 0);
  }

  function fail(msg) {
    stopTimers();
    $("error-msg").textContent = msg;
    show("error");
  }

  function toSearch() {
    stopTimers();
    history.replaceState(null, "", location.pathname);
    document.title = "peakweek";
    $("places").hidden = true;
    setMsg("");
    show("search");
    $("q").focus();
  }

  // ---- buttons
  $("retry").addEventListener("click", () => (current ? start(current) : toSearch()));
  $("error-back").addEventListener("click", toSearch);
  $("again").addEventListener("click", toSearch);
  $("print").addEventListener("click", () => window.print());
  $("done").addEventListener("click", () => { show("outside"); window.scrollTo(0, 0); });
  $("back").addEventListener("click", () => { show("result"); window.scrollTo(0, 0); });

  // ---- boot: ?lat=&lon=&name= reopens a card (bookmarkable)
  loadInfo().then(() => {
    const p = new URLSearchParams(location.search);
    const lat = parseFloat(p.get("lat")), lon = parseFloat(p.get("lon"));
    if (Number.isFinite(lat) && Number.isFinite(lon)) start({ lat, lon, name: p.get("name") || "" });
    else show("search");
  });
})();
