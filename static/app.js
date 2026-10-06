// peakweek page: search -> forecast job (polled) -> card. The card HTML is rendered (and escaped)
// by the server's peakweek/card.py; this script only moves between views and fills in plain text.
"use strict";
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

  const fmtInt = (n) => Number(n).toLocaleString("en-US");
  const round4 = (x) => Math.round(x * 1e4) / 1e4;
  function setMsg(text) { $("search-msg").textContent = text || ""; }

  async function loadInfo() {
    try { info = await getJSON("/api/info"); } catch (_) { return; }
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
      const data = await getJSON("/api/geocode?q=" + encodeURIComponent(q));
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
      handle(await getJSON("/api/forecast?" + params.toString()));
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
        handle(await getJSON("/api/forecast/status?id=" + encodeURIComponent(job.id)));
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
    if (mode === "fixture") {
      what = "Loading the saved example. (A live forecast means " + steps + ".)";
    } else if (job && job.status === "queued" && job.ahead > 0) {
      what = "Waiting for " + (job.ahead === 1 ? "another forecast" : job.ahead + " other forecasts") +
        " to finish first (one at a time). Then: " + steps + "…";
    } else {
      what = steps.charAt(0).toUpperCase() + steps.slice(1) + "…";
    }
    $("wait-what").textContent = what;
    let time = secs + " s so far";
    if (e.seconds && mode !== "fixture") time += " · a recent run took " + Math.round(e.seconds) + " s";
    $("wait-time").textContent = time;
  }

  function finish(job) {
    stopTimers();
    $("card").innerHTML = job.card_html;  // rendered and escaped server-side (peakweek/card.py)
    const r = job.result || {};
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
