// Record the peakweek demo: drives a throwaway headless Chrome over the DevTools protocol, saves clean
// screenshots to docs/screenshots/ (desktop 1280 and phone 390) and builds docs/demo.gif from screencast frames.
//
//   python3 server.py --fixture examples/forecast_fixture.json --fixture-delay 8   (or the live server)
//   node --experimental-websocket scripts/record_demo.mjs
//
// Env: APP_URL (default http://127.0.0.1:8770/), PLACE (typed on desktop, default "New Brunswick, NJ"),
//      GEO ("lat,lon" the phone pass reports for "Use my location", default Rutgers Gardens 40.4741,-74.4228),
//      WAIT_SPEEDUP (how much faster the waiting part plays in the GIF, default 4; ~10 for a live run).
// The fake cursor and captions are injected by this script for the video only; they are not part of the app.
// "Print card" is shown by emulating print media (window.print is stubbed in the recorded page so headless
// Chrome does not block); docs/screenshots/print-card.pdf is the real print output.
import { spawn, execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const APP = process.env.APP_URL || "http://127.0.0.1:8770/";
const PLACE = process.env.PLACE || "New Brunswick, NJ";
const [GEO_LAT, GEO_LON] = (process.env.GEO || "40.4741,-74.4228").split(",").map(Number);
const SPEEDUP = Number(process.env.WAIT_SPEEDUP || 4);
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9333;
const DESK = { width: 1280, height: 800, deviceScaleFactor: 2, mobile: false };
const PHONE = { width: 390, height: 844, deviceScaleFactor: 2, mobile: true };
const LETTER = { width: 816, height: 1056, deviceScaleFactor: 2, mobile: false }; // 8.5 x 11 in at 96 dpi
const work = mkdtempSync(join(tmpdir(), "peakweek-rec-"));
const framesDir = join(work, "frames");
mkdirSync(framesDir);
const shotsDir = join(ROOT, "docs", "screenshots");
mkdirSync(shotsDir, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--disable-extensions",
  "--use-mock-keychain", "--password-store=basic", "--hide-scrollbars", "--mute-audio",
  `--user-data-dir=${join(work, "profile")}`, `--remote-debugging-port=${PORT}`,
  `--window-size=${DESK.width},${DESK.height}`, "about:blank",
], { stdio: "ignore" });

let ws, nextId = 1;
const pending = new Map();
const frames = [];
let recording = false, speedup = 1, cut = false;
// Pause the GIF without leaving a frozen gap: the first frame after a pause starts a new cut.
function setRecording(on) { if (on && !recording) cut = true; recording = on; }

// Close Chrome over the DevTools protocol so it deletes its code-sign clone of the app bundle
// ($TMPDIR/../X/com.google.Chrome.code_sign_clone, ~1.4 GB each); SIGKILL and SIGTERM leave the clone behind.
async function closeChrome() {
  if (!chrome || chrome.exitCode !== null || chrome.signalCode !== null) return;
  const exited = new Promise((r) => chrome.once("exit", () => r(true)));
  for (let i = 0; i < 20; i++) {
    try {
      const { webSocketDebuggerUrl } = await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json();
      const b = new WebSocket(webSocketDebuggerUrl);
      b.onopen = () => b.send(JSON.stringify({ id: 1, method: "Browser.close" }));
      break;
    } catch { await sleep(250); }
  }
  if (!(await Promise.race([exited, sleep(5000).then(() => false)]))) chrome.kill("SIGKILL");
}

async function connect() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch {}
    await sleep(200);
  }
  throw new Error("Chrome did not start");
}
function send(method, params = {}) {
  const id = nextId++;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}
async function js(expr) {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || "eval failed");
  return r.result.value;
}

const CURSOR_JS = `(() => {
  if (document.getElementById('__cur')) return;
  const c = document.createElement('div'); c.id='__cur';
  c.innerHTML = '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M4 2l15 9.5-6.6 1.3 3.9 7.4-2.7 1.4-3.9-7.4L4 19z" fill="#111" stroke="#fff" stroke-width="1.5"/></svg>';
  Object.assign(c.style,{position:'fixed',left:'0',top:'0',zIndex:99999,pointerEvents:'none',transition:'transform .55s cubic-bezier(.3,.7,.3,1)',transform:'translate(640px,420px)'});
  document.body.appendChild(c);
  const cap = document.createElement('div'); cap.id='__cap';
  Object.assign(cap.style,{position:'fixed',left:'50%',bottom:'18px',transform:'translateX(-50%)',zIndex:99998,pointerEvents:'none',
    background:'rgba(20,20,18,.88)',color:'#fff',font:'600 15px -apple-system,Helvetica,sans-serif',padding:'9px 16px',borderRadius:'10px',maxWidth:'88%',textAlign:'center',transition:'opacity .3s',opacity:0});
  document.body.appendChild(cap);
})()`;
const caption = (t) => js(`(()=>{const c=document.getElementById('__cap'); if(!c) return; c.textContent=${JSON.stringify(t)}; c.style.opacity=${t ? 1 : 0};})()`);
const overlays = (on) => js(`['__cur','__cap'].forEach(id=>{const e=document.getElementById(id); if(e) e.style.visibility=${on ? "'visible'" : "'hidden'"}})`);

async function moveTo(x, y) {
  await js(`(()=>{const c=document.getElementById('__cur'); if(c) c.style.transform='translate(${x - 3}px,${y - 2}px)'})()`);
  await sleep(650);
}
async function center(sel) {
  return js(`(()=>{const e=document.querySelector(${JSON.stringify(sel)}); e.scrollIntoView({block:'nearest'});
    const r=e.getBoundingClientRect(); return {x:r.left+r.width/2, y:r.top+r.height/2};})()`);
}
async function mouse(type, x, y, buttons = 1) {
  await send("Input.dispatchMouseEvent", { type, x, y, button: "left", buttons, clickCount: 1 });
}
async function click(sel) {
  const p = await center(sel);
  await moveTo(p.x, p.y);
  await mouse("mousePressed", p.x, p.y);
  await sleep(60);
  await mouse("mouseReleased", p.x, p.y, 0);
  await sleep(250);
}
async function typeText(text) {
  for (const ch of text) {
    await send("Input.insertText", { text: ch });
    await sleep(55);
  }
}
async function smoothScroll(top, ms = 1200) {
  await js(`window.scrollTo({top:${top}, behavior:'smooth'})`);
  await sleep(ms);
}
async function topOf(sel, offset = 0) {
  return js(`(()=>{const e=document.querySelector(${JSON.stringify(sel)}); return e.getBoundingClientRect().top + scrollY - ${offset}})()`);
}
async function shot(name, { full = false } = {}) {
  const wasRecording = recording;
  if (full) setRecording(false); // a full-page capture resizes the viewport; keep those frames out of the GIF
  await overlays(false);
  await sleep(150);
  const params = { format: "png" };
  if (full) {
    const { cssContentSize: s } = await send("Page.getLayoutMetrics");
    Object.assign(params, { captureBeyondViewport: true, clip: { x: 0, y: 0, width: s.width, height: s.height, scale: 1 } });
  }
  const r = await send("Page.captureScreenshot", params);
  writeFileSync(join(shotsDir, name), Buffer.from(r.data, "base64"));
  await overlays(true);
  if (full) { await sleep(200); setRecording(wasRecording); }
  console.log("screenshot", name);
}
async function visible(id) {
  return js(`(()=>{const e=document.getElementById(${JSON.stringify(id)}); return !!e && !e.hidden})()`);
}
async function waitFor(expr, timeoutMs = 300000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (await js(expr)) return Date.now() - t0;
    await sleep(250);
  }
  throw new Error("timeout waiting for " + expr);
}
const shown = (id) => `(()=>{const e=document.getElementById(${JSON.stringify(id)}); return !!e && !e.hidden})()`;
async function errorText() {
  return (await visible("error")) ? js("document.getElementById('error-msg').textContent") : null;
}
async function setViewport(m, scheme = "light", media = "") {
  await send("Emulation.setDeviceMetricsOverride", m);
  await send("Emulation.setTouchEmulationEnabled", { enabled: !!m.mobile });
  await send("Emulation.setEmulatedMedia", { media, features: [{ name: "prefers-color-scheme", value: scheme }] });
}
async function open(url) {
  await send("Page.navigate", { url });
  await waitFor("document.readyState === 'complete' && !!document.getElementById('search')", 20000);
  await sleep(600);
}
function pdfPages(buf) {
  return (buf.toString("latin1").match(/\/Type\s*\/Page(?![s\w])/g) || []).length;
}

async function desktopPass() {
  await setViewport(DESK);
  await open(APP);
  await js(CURSOR_JS);
  await js("window.print = () => { window.__printRequested = true; }"); // recorder-only stub, see header
  await send("Page.startScreencast", { format: "jpeg", quality: 85, maxWidth: DESK.width, maxHeight: DESK.height, everyNthFrame: 1 });
  setRecording(true);

  await caption("Which trees near you are turning this weekend? Type a town");
  await sleep(1400);
  await click("#q");
  await typeText(PLACE);
  await sleep(300);
  await click("#search-form button[type=submit]");
  await waitFor(`${shown("places")} || ${shown("waiting")}`, 20000);
  await sleep(700);
  await shot("desktop-1-search.png");
  if (await visible("places")) {
    await caption("Pick the place");
    await sleep(900);
    await click("#places .place-btn");
  }
  await waitFor(`${shown("waiting")} || ${shown("result")} || ${shown("error")}`, 20000);
  if (await visible("waiting")) {
    speedup = SPEEDUP;
    await caption(`Forecast running: weather, then TabPFN (this wait sped up ${SPEEDUP}x)`);
    await sleep(1600);
    await shot("desktop-2-waiting.png");
  }
  const waited = await waitFor(`${shown("result")} || ${shown("error")}`);
  speedup = 1;
  const err = await errorText();
  if (err) throw new Error(err);
  console.log(`waited ${(waited / 1000).toFixed(1)} s for the forecast`);

  await caption("One card: which trees to see this weekend, and when to come back");
  await sleep(2800);
  await shot("desktop-3-card.png");
  await shot("desktop-3-card-full.png", { full: true });
  await smoothScroll(await topOf(".sp", 90), 1400);
  await caption("Chance a tree you find shows color, its best day, and what its leaves look like");
  await sleep(3200);
  await smoothScroll(await topOf(".sp:nth-child(3)", 90), 1600);
  await caption("The small strip is the next 15 days; the weekend is shaded");
  await sleep(2600);
  await smoothScroll(await topOf(".actions", 520), 1600);
  await sleep(600);

  await caption("Print card: one black-and-white page to take along");
  await click("#print");
  await send("Emulation.setEmulatedMedia", { media: "print", features: [{ name: "prefers-color-scheme", value: "light" }] });
  await smoothScroll(0, 600);
  await sleep(2600);
  await smoothScroll(await js("document.body.scrollHeight"), 1800);
  await sleep(800);

  setRecording(false); // the letter-size preview and PDF are not part of the GIF
  await send("Emulation.setDeviceMetricsOverride", LETTER);
  // Preview only: pad the body like the @page margins so lines wrap as on paper; removed before the PDF.
  await js(`(()=>{const s=document.createElement('style'); s.id='__page';
    s.textContent='@media print{body{padding:11mm 12mm !important}}'; document.head.appendChild(s)})()`);
  await sleep(500);
  await shot("4-print-preview.png", { full: true });
  await js("document.getElementById('__page').remove()");
  const pdf = await send("Page.printToPDF", { printBackground: true, paperWidth: 8.5, paperHeight: 11, preferCSSPageSize: true });
  const buf = Buffer.from(pdf.data, "base64");
  writeFileSync(join(shotsDir, "print-card.pdf"), buf);
  console.log(`print-card.pdf: ${pdfPages(buf)} page(s)`);
  await setViewport(DESK);
  await smoothScroll(await topOf(".actions", 520), 300);
  await sleep(400);
  setRecording(true);

  await caption("Done: the page shrinks to one sentence");
  await sleep(900);
  await click("#done");
  await sleep(1800);
  await shot("desktop-5-outside.png");
  await caption("Screen off. Go outside.");
  await sleep(2400);
  await caption("");
  await sleep(500);
  setRecording(false);
  await send("Page.stopScreencast");
}

async function phonePass() {
  const origin = new URL(APP).origin;
  await send("Browser.grantPermissions", { origin, permissions: ["geolocation"] });
  await send("Emulation.setGeolocationOverride", { latitude: GEO_LAT, longitude: GEO_LON, accuracy: 50 });
  await setViewport(PHONE);
  await open(APP);
  await shot("phone-1-search.png");
  await click("#locate");
  await waitFor(`${shown("waiting")} || ${shown("result")} || ${shown("error")}`, 30000);
  if (await visible("waiting")) {
    await sleep(1600);
    await shot("phone-2-waiting.png");
  } else {
    console.log("phone: no waiting state (result was cached)");
  }
  await waitFor(`${shown("result")} || ${shown("error")}`);
  const err = await errorText();
  if (err) throw new Error(err);
  await sleep(800);
  await shot("phone-3-card.png");
  await shot("phone-3-card-full.png", { full: true });
  await setViewport(PHONE, "dark");
  await sleep(500);
  await shot("phone-3-card-dark.png");
  await setViewport(PHONE);
  await js("document.getElementById('done').click()");
  await sleep(700);
  await shot("phone-5-outside.png");
}

function assemble() {
  // Re-time frames: real gaps, divided by the speed-up active when the frame arrived; cap long idle gaps.
  const list = [];
  let videoT = 0;
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const next = frames[i + 1];
    let dur = next ? (next.t - f.t) / f.speedup : 1.0;
    dur = Math.min(Math.max(dur, 0.001), next && next.cut ? 0.15 : 2.5);
    const file = join(framesDir, `f${String(i).padStart(5, "0")}.jpg`);
    writeFileSync(file, Buffer.from(f.data, "base64"));
    list.push(`file '${file}'\nduration ${dur.toFixed(4)}`);
    videoT += dur;
  }
  list.push(`file '${join(framesDir, `f${String(frames.length - 1).padStart(5, "0")}.jpg`)}'`);
  const concat = join(work, "frames.txt");
  writeFileSync(concat, list.join("\n") + "\n");
  const gif = join(ROOT, "docs", "demo.gif");
  execFileSync("nice", ["-n", "19", "ffmpeg", "-y", "-loglevel", "error", "-threads", "1", "-f", "concat", "-safe", "0", "-i", concat,
    "-vf", "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
    "-threads", "1", gif]);
  console.log(`frames ${frames.length}, gif ${videoT.toFixed(1)} s ->`, gif);
  // Same frames as an MP4 (H.264, 1280 wide) for the "full video" link.
  const mp4 = join(ROOT, "docs", "demo.mp4");
  execFileSync("nice", ["-n", "19", "ffmpeg", "-y", "-loglevel", "error", "-threads", "1", "-f", "concat", "-safe", "0", "-i", concat,
    "-vf", "fps=25,scale=1280:-2:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
    "-movflags", "+faststart", "-threads", "1", mp4]);
  console.log("mp4 ->", mp4);
}

async function main() {
  ws = new WebSocket(await connect());
  await new Promise((r) => (ws.onopen = r));
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const p = pending.get(msg.id); pending.delete(msg.id);
      msg.error ? p.reject(new Error(msg.error.message)) : p.resolve(msg.result);
    } else if (msg.method === "Page.screencastFrame") {
      const { data, sessionId, metadata } = msg.params;
      // Keep only frames at the recorded viewport size (deviceWidth/Height are CSS pixels).
      if (recording && Math.round(metadata.deviceWidth) === DESK.width && Math.round(metadata.deviceHeight) === DESK.height) {
        frames.push({ t: metadata.timestamp, data, speedup, cut });
        cut = false;
      }
      send("Page.screencastFrameAck", { sessionId }).catch(() => {});
    }
  };
  await send("Page.enable");
  await send("Runtime.enable");
  await desktopPass();
  assemble();
  await phonePass();
}

try {
  await main();
} catch (e) {
  console.error("FAILED:", e.message);
  process.exitCode = 1;
} finally {
  try { ws?.close(); } catch {}
  await closeChrome();
  await sleep(300);
  if (process.env.KEEP_WORK) console.log("kept", work);
  else rmSync(work, { recursive: true, force: true });
}
