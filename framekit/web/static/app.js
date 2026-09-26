"use strict";
/* Shared helpers for every page: fetch wrapper, formatting, toasts, and the
   jobs panel. Pages wait for a job with FK.waitJob(id) -> Promise<job>. */
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const FK = {
  async api(path, opts = {}) {
    if (opts.json !== undefined) {
      opts = {...opts, method: opts.method || "POST", headers: {"Content-Type": "application/json"},
              body: JSON.stringify(opts.json)};
      delete opts.json;
    }
    const r = await fetch(path, opts);
    const ct = r.headers.get("content-type") || "";
    const data = ct.includes("json") ? await r.json() : await r.text();
    if (!r.ok) throw new Error((data && data.error) || data || r.statusText);
    return data;
  },
  bytes(n) {
    if (!n) return "0 B";
    const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(n < 10 && i ? 1 : 0)} ${u[i]}`;
  },
  dur(s) {
    if (s == null) return "";
    s = Math.round(s);
    const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), ss = s % 60;
    return (h ? `${h}:${String(m).padStart(2, "0")}` : `${m}`) + `:${String(ss).padStart(2, "0")}`;
  },
  when(t) { return t ? new Date(t * 1000).toLocaleString() : ""; },
  toast(msg, bad = false) {
    let t = $("toast");
    if (!t) { t = document.createElement("div"); t.id = "toast"; document.body.append(t); }
    t.textContent = msg; t.className = bad ? "bad" : ""; t.hidden = false;
    clearTimeout(t._h); t._h = setTimeout(() => t.hidden = true, bad ? 7000 : 3500);
  },
  fail(e) { FK.toast(e.message || String(e), true); },

  /* ---- jobs ---- */
  _waiters: {},
  _jobs: [],
  waitJob(id) { return new Promise((res, rej) => { FK._waiters[id] = {res, rej}; FK.pollJobs(); }); },
  async pollJobs() {
    let list;
    try { list = await FK.api("/api/jobs"); } catch { return; }
    FK._jobs = list;
    for (const j of list) {
      const w = FK._waiters[j.id];
      if (!w || !j.finished) continue;
      delete FK._waiters[j.id];
      if (j.state === "done") w.res(j); else w.rej(new Error(j.error || j.state));
    }
    FK.renderJobs();
  },
  renderJobs() {
    const active = FK._jobs.filter(j => !j.finished);
    const badge = $("jobsBadge");
    if (badge) { badge.textContent = active.length; badge.hidden = !active.length; }
    const box = $("jobsList");
    if (!box || $("jobs").hidden) return;
    box.innerHTML = FK._jobs.length ? "" : '<div class="dim">No jobs yet.</div>';
    for (const j of FK._jobs.slice(0, 40)) {
      const d = document.createElement("div");
      d.className = "job";
      const pct = j.total ? Math.round(100 * j.progress / j.total) : null;
      d.innerHTML = `<div class="row"><b class="grow">${esc(j.label)}</b>
          <span class="${j.state === "error" ? "err" : j.state === "done" ? "ok" : "dim"}">${j.state}</span></div>
        <div class="dim">${esc(j.message || "")}${pct != null && !j.finished ? ` · ${pct}%` : ""}</div>
        ${j.error ? `<div class="err">${esc(j.error)}</div>` : ""}
        ${!j.finished ? `<progress max="${j.total || 1}" ${j.total ? `value="${j.progress}"` : ""}></progress>` : ""}`;
      if (!j.finished) {
        const b = document.createElement("button");
        b.className = "small danger"; b.textContent = "Cancel";
        b.onclick = () => FK.api(`/api/jobs/${j.id}/cancel`, {method: "POST"}).then(FK.pollJobs);
        d.querySelector(".row").append(b);
      }
      box.append(d);
    }
  },
  /* Submit a job-returning request, show progress in `bar` (a <progress>), resolve when done. */
  async runJob(path, body, bar, label) {
    const {job} = await FK.api(path, {json: body || {}});
    if (bar) { bar.hidden = false; bar.removeAttribute("value"); }
    const tick = setInterval(() => {
      const j = FK._jobs.find(x => x.id === job);
      if (j && bar) {
        if (j.total) { bar.max = j.total; bar.value = j.progress; } else bar.removeAttribute("value");
        if (label) label.textContent = `${j.message || ""}${j.total ? ` ${j.progress}/${j.total}` : ""}`;
      }
    }, 500);
    try { return await FK.waitJob(job); }
    finally { clearInterval(tick); if (bar) bar.hidden = true; if (label) label.textContent = ""; }
  },
};

document.addEventListener("DOMContentLoaded", () => {
  const btn = $("jobsBtn");
  if (btn) btn.onclick = () => { $("jobs").hidden = !$("jobs").hidden; FK.renderJobs(); };
  FK.pollJobs();
  setInterval(FK.pollJobs, 1000);
});
