"""The workspace page, embedded as a string.

Everything is inline on purpose. A tool whose entire promise is "works with the
network unplugged" cannot fetch a stylesheet, a font or a chart library from a
CDN -- the first offline demo would render as unstyled text. So: system font
stacks, hand-written CSS, no build step, one file.

Visual language -- "instrument", not "dashboard". The reference is a lab
notebook next to a bench meter: warm grey-green paper, dark ink, one saturated
green for measured values and one amber for anything the operator must not
mistake for a measurement.

The signature element is the **inference trace rail**: the ten DIC phases drawn
as a vertical rail down the right side, each with its real duration. It is the
only place in the product where the architecture is directly visible to the
user, and it is what makes the determinism claim legible rather than asserted.
"""
from __future__ import annotations

PAGE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeanLM -- atelier local</title>
<style>
:root{
  --paper:#EEF1EE; --paper-2:#E4E9E4; --card:#F7F9F7;
  --ink:#16211D; --ink-2:#42514B; --ink-3:#7A8A83;
  --rule:#CBD5CE; --rule-2:#DCE3DD;
  --green:#0F6E4F; --green-soft:#E1EFE8;
  --amber:#B26B00; --amber-soft:#F6ECD9;
  --red:#8C2F1E; --red-soft:#F4E2DE;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"DejaVu Sans Mono",monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--paper); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
}
a{color:var(--green)}
button{font-family:inherit;font-size:inherit}

/* ---------- masthead ---------- */
header{
  border-bottom:1px solid var(--rule); background:var(--paper);
  position:sticky; top:0; z-index:20;
}
.masthead{
  max-width:1180px; margin:0 auto; padding:14px 24px;
  display:flex; align-items:baseline; gap:18px; flex-wrap:wrap;
}
.brand{font-weight:650; letter-spacing:-.01em; font-size:17px}
.brand span{color:var(--green)}
.tagline{color:var(--ink-3); font-size:13px; flex:1; min-width:200px}
.chips{display:flex; gap:6px; flex-wrap:wrap}
.chip{
  font-family:var(--mono); font-size:11px; letter-spacing:.02em;
  padding:3px 8px; border:1px solid var(--rule); border-radius:2px;
  background:var(--card); color:var(--ink-2); white-space:nowrap;
}
.chip.on{border-color:var(--green); color:var(--green); background:var(--green-soft)}
.chip.warn{border-color:var(--amber); color:var(--amber); background:var(--amber-soft)}
.chip.bad{border-color:var(--red); color:var(--red); background:var(--red-soft)}

.modes{display:flex; border:1px solid var(--rule); border-radius:2px; overflow:hidden}
.modes button{
  background:var(--card); border:0; padding:5px 12px; cursor:pointer;
  color:var(--ink-2); font-size:12px; letter-spacing:.02em;
}
.modes button + button{border-left:1px solid var(--rule)}
.modes button[aria-pressed="true"]{background:var(--ink); color:var(--paper)}

/* ---------- layout ---------- */
main{max-width:1180px; margin:0 auto; padding:26px 24px 80px;
     display:grid; grid-template-columns:minmax(0,1fr) 268px; gap:34px}
@media (max-width:900px){main{grid-template-columns:minmax(0,1fr)}}

section{margin-bottom:26px}
h2{font-size:12px; text-transform:uppercase; letter-spacing:.10em;
   color:var(--ink-3); margin:0 0 10px; font-weight:600}

/* ---------- ask ---------- */
.ask{display:flex; gap:10px; align-items:stretch}
.ask textarea{
  flex:1; resize:vertical; min-height:62px; padding:11px 13px;
  border:1px solid var(--rule); border-radius:3px; background:var(--card);
  font-family:inherit; font-size:15px; color:var(--ink);
}
.ask textarea:focus{outline:2px solid var(--green); outline-offset:-1px}
.ask button{
  padding:0 20px; border:1px solid var(--green); background:var(--green);
  color:#fff; border-radius:3px; cursor:pointer; font-weight:600; letter-spacing:.01em;
}
.ask button:disabled{opacity:.45; cursor:progress}
.samples{margin-top:9px; display:flex; gap:7px; flex-wrap:wrap}
.samples button{
  border:1px dashed var(--rule); background:transparent; color:var(--ink-2);
  padding:3px 9px; border-radius:2px; cursor:pointer; font-size:12.5px;
}
.samples button:hover{border-color:var(--green); color:var(--green)}

/* ---------- answer ---------- */
.answer{
  background:var(--card); border:1px solid var(--rule); border-left:3px solid var(--green);
  border-radius:3px; padding:18px 20px; white-space:pre-wrap;
}
.answer.empty{color:var(--ink-3); border-left-color:var(--rule); white-space:normal}
.banner{
  background:var(--amber-soft); border:1px solid var(--amber); color:#6b4100;
  padding:9px 13px; border-radius:3px; margin-bottom:12px; font-size:13.5px;
}
.banner.bad{background:var(--red-soft); border-color:var(--red); color:var(--red)}

.sources{list-style:none; margin:12px 0 0; padding:0}
.sources li{
  border-top:1px solid var(--rule-2); padding:9px 0;
  display:grid; grid-template-columns:34px 1fr auto; gap:10px; align-items:baseline;
}
.sources .label{font-family:var(--mono); font-size:12px; color:var(--green); font-weight:600}
.sources .where{font-size:12.5px; color:var(--ink-3)}
.sources .score{font-family:var(--mono); font-size:11.5px; color:var(--ink-3)}
.sources .excerpt{grid-column:2/4; color:var(--ink-2); font-size:13.5px; margin-top:3px}

/* ---------- verdict ---------- */
.verdict{display:flex; gap:9px; flex-wrap:wrap; margin-top:14px}
.badge{
  font-size:12px; padding:4px 10px; border-radius:2px; border:1px solid var(--rule);
  background:var(--card); color:var(--ink-2);
}
.badge b{font-family:var(--mono); font-weight:600}
.badge.ok{border-color:var(--green); background:var(--green-soft); color:var(--green)}
.badge.warn{border-color:var(--amber); background:var(--amber-soft); color:var(--amber)}
.badge.bad{border-color:var(--red); background:var(--red-soft); color:var(--red)}

/* ---------- trace rail (signature) ---------- */
.rail{position:sticky; top:78px}
.rail ol{list-style:none; margin:0; padding:0; position:relative}
.rail ol::before{
  content:""; position:absolute; left:5px; top:6px; bottom:6px;
  width:1px; background:var(--rule);
}
.rail li{position:relative; padding:0 0 0 20px; margin-bottom:9px}
.rail li::before{
  content:""; position:absolute; left:1px; top:6px; width:9px; height:9px;
  border-radius:50%; background:var(--paper); border:1.5px solid var(--rule);
}
.rail li.done::before{background:var(--green); border-color:var(--green)}
.rail li.failed::before{background:var(--red); border-color:var(--red)}
.rail .name{font-size:11.5px; color:var(--ink-2); letter-spacing:.01em}
.rail li.done .name{color:var(--ink)}
.rail .ms{font-family:var(--mono); font-size:11px; color:var(--ink-3)}
.rail .bar{height:3px; background:var(--rule-2); border-radius:2px; margin-top:3px}
.rail .bar i{display:block; height:100%; background:var(--green); border-radius:2px}
.rail li.failed .bar i{background:var(--red)}
.rail .total{
  margin-top:14px; padding-top:10px; border-top:1px solid var(--rule);
  font-family:var(--mono); font-size:11.5px; color:var(--ink-2);
}

/* ---------- engineering ---------- */
.proof{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(104px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); border-radius:3px;
  margin-top:2px;
}
.proof .cell{background:var(--paper-raised); padding:11px 12px; text-align:left}
.proof .k{
  font-family:var(--mono); font-size:9px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--ink-soft);
}
.proof .v{font-family:var(--mono); font-size:20px; font-variant-numeric:tabular-nums;
  line-height:1.2; margin-top:3px}
.proof .u{font-family:var(--mono); font-size:10px; color:var(--ink-soft)}
.proof .cell.offline .v{color:var(--instrument)}
.proof .cell.offline.bad .v{color:var(--alert)}
.eng{display:none}
body.engineering .eng{display:block}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(148px,1fr)); gap:1px;
      background:var(--rule-2); border:1px solid var(--rule-2)}
.cell{background:var(--card); padding:10px 12px}
.cell .k{font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-3)}
.cell .v{font-family:var(--mono); font-size:17px; color:var(--ink); margin-top:2px}
.cell .u{font-size:11px; color:var(--ink-3)}
.cell.muted .v{color:var(--ink-3)}
details{border:1px solid var(--rule); border-radius:3px; background:var(--card); margin-top:12px}
summary{cursor:pointer; padding:9px 13px; font-size:13px; color:var(--ink-2)}
details pre{
  margin:0; padding:0 13px 13px; overflow:auto; max-height:340px;
  font-family:var(--mono); font-size:11.5px; color:var(--ink-2); line-height:1.5;
}
table.kv{border-collapse:collapse; width:100%; font-size:13px}
table.kv td{border-top:1px solid var(--rule-2); padding:6px 4px; vertical-align:top}
table.kv td:first-child{color:var(--ink-3); width:44%}
table.kv td:last-child{font-family:var(--mono); font-size:12px}

/* ---------- corpus ---------- */
.docs{border:1px solid var(--rule-2); border-radius:3px; overflow:hidden}
.docs .row{display:grid; grid-template-columns:1fr 72px 84px; gap:8px;
           padding:8px 12px; background:var(--card); font-size:13px}
.docs .row + .row{border-top:1px solid var(--rule-2)}
.docs .row.head{background:var(--paper-2); color:var(--ink-3); font-size:11px;
                text-transform:uppercase; letter-spacing:.06em}
.docs .num{font-family:var(--mono); font-size:12px; text-align:right}
.hint{color:var(--ink-3); font-size:13px}
footer{border-top:1px solid var(--rule); color:var(--ink-3); font-size:12px;
       padding:14px 24px; max-width:1180px; margin:0 auto}
.spin{display:inline-block; width:9px; height:9px; border:2px solid var(--rule);
      border-top-color:var(--green); border-radius:50%; animation:s .8s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header>
  <div class="masthead">
    <div class="brand">Lean<span>LM</span></div>
    <div class="tagline">Couche d'optimisation d'inference locale &mdash; tout se passe sur cette machine.</div>
    <div class="chips" id="chips"></div>
    <div class="modes">
      <button id="mode-demo" aria-pressed="true">Demonstration</button>
      <button id="mode-eng" aria-pressed="false">Ingenierie</button>
    </div>
  </div>
</header>

<main>
  <div>
    <section>
      <h2>Question</h2>
      <div class="ask">
        <textarea id="q" placeholder="Posez une question sur les documents du corpus local&hellip;"></textarea>
        <button id="send">Interroger</button>
      </div>
      <div class="samples" id="samples"></div>
    </section>

    <section>
      <h2>Reponse</h2>
      <div id="banners"></div>
      <div class="answer empty" id="answer">Aucune requete pour l'instant. La reponse s'appuiera uniquement sur les extraits retenus dans le corpus local.</div>
      <ul class="sources" id="sources"></ul>
      <div class="verdict" id="verdict"></div>
    </section>

    <section>
      <h2>La preuve</h2>
      <div class="proof" id="proof"></div>
      <p class="hint">Rien de tout cela ne quitte cette machine. Aucune connexion
      sortante n'est autorisee pendant l'inference : une tentative echoue et est
      enregistree.</p>
    </section>

    <section class="eng">
      <h2>Mesures de l'execution</h2>
      <div class="grid" id="metrics"></div>
      <details><summary>Budget de contexte et decisions</summary><div style="padding:0 13px 13px"><table class="kv" id="budget"></table></div></details>
      <details><summary>Validation detaillee</summary><div style="padding:0 13px 13px"><table class="kv" id="valdetail"></table></div></details>
      <details><summary>Contexte d'execution complet (JSON)</summary><pre id="raw"></pre></details>
    </section>

    <section>
      <h2>Corpus local</h2>
      <div class="docs" id="docs"></div>
      <p class="hint">Les documents ne sont jamais copies ni modifies : seul leur contenu segmente est indexe dans <code>.leanlm/corpus.sqlite3</code>. Ajoutez-en avec <code>leanlm ingest &lt;chemin&gt;</code>.</p>
    </section>
  </div>

  <aside>
    <div class="rail">
      <h2>Trace d'inference</h2>
      <ol id="rail"></ol>
      <div class="total" id="railtotal">en attente</div>
      <p class="hint" style="margin-top:12px">Les dix phases du contrat deterministe, dans l'ordre, avec leur duree reelle.</p>
    </div>
  </aside>
</main>

<footer id="foot"></footer>

<script>
const STAGES = [
  ["session_creation","Session"],
  ["resource_assessment","Ressources"],
  ["document_discovery","Corpus"],
  ["context_optimization","Optimisation"],
  ["evidence_retrieval","Recherche"],
  ["prompt_assembly","Prompt"],
  ["model_inference","Inference"],
  ["response_validation","Validation"],
  ["metrics_finalization","Metriques"],
  ["session_cleanup","Nettoyage"]
];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s)
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const num = (v, d = 1) => (v === null || v === undefined) ? "-" : Number(v).toFixed(d);

let STATUS = null;

/* ---------- mode ---------- */
function setMode(engineering){
  document.body.classList.toggle("engineering", engineering);
  $("mode-eng").setAttribute("aria-pressed", String(engineering));
  $("mode-demo").setAttribute("aria-pressed", String(!engineering));
  try { localStorage.setItem("leanlm.mode", engineering ? "eng" : "demo"); } catch (e) {}
}
$("mode-eng").onclick = () => setMode(true);
$("mode-demo").onclick = () => setMode(false);
try { setMode(localStorage.getItem("leanlm.mode") === "eng"); } catch (e) { setMode(false); }

/* ---------- status ---------- */
async function loadStatus(){
  const res = await fetch("/api/status");
  STATUS = await res.json();
  const p = STATUS.profile || {}, b = STATUS.backend || {}, c = STATUS.corpus || {};
  const chips = [
    ["profil " + (p.id || "?"), "on"],
    ["moteur " + (b.backend || "?"), b.is_simulated ? "warn" : "on"],
    ["modele " + ((p.model && p.model.id) || "?"), "on"],
    [c.documents + " doc / " + c.units + " unites", c.documents ? "on" : "warn"],
    [STATUS.offline_enforced ? "hors ligne applique" : "hors ligne non applique",
     STATUS.offline_enforced ? "on" : "bad"]
  ];
  if (b.is_simulated) chips.push(["backend simule", "warn"]);
  $("chips").innerHTML = chips
    .map(([t, k]) => '<span class="chip ' + k + '">' + esc(t) + "</span>").join("");
  renderProof(null);

  const r = STATUS.resources || {};
  const t = STATUS.trust && STATUS.trust.runtime && STATUS.trust.runtime.details || {};
  $("foot").innerHTML =
    "RAM disponible " + num(r.available_ram_mb, 0) + " Mo &middot; CPU " +
    num(r.cpu_percent, 0) + " % &middot; temperature " +
    (r.temperature_c === null || r.temperature_c === undefined
      ? "non mesuree (" + esc(r.thermal_source || "aucun capteur") + ")"
      : num(r.temperature_c, 1) + " &deg;C") +
    " &middot; empreinte runtime " + esc(t.source_fingerprint || "?") +
    " &middot; profil " + esc(p.fingerprint || "?");

  renderDocs(STATUS.documents || []);
  renderSamples(STATUS.suggested_questions || []);
  renderRail(null);
}

function renderDocs(docs){
  if (!docs.length){
    $("docs").innerHTML = '<div class="row"><span class="hint">Corpus vide.</span></div>';
    return;
  }
  const head = '<div class="row head"><span>document</span><span class="num">unites</span><span class="num">tokens</span></div>';
  $("docs").innerHTML = head + docs.map((d) =>
    '<div class="row"><span>' + esc(d.filename) + '</span><span class="num">' +
    d.unit_count + '</span><span class="num">' + d.token_estimate + "</span></div>"
  ).join("");
}

function renderSamples(list){
  $("samples").innerHTML = list.map((q) =>
    "<button>" + esc(q) + "</button>").join("");
  Array.from($("samples").children).forEach((b) => {
    b.onclick = () => { $("q").value = b.textContent; ask(); };
  });
}

/* ---------- trace rail ---------- */
function renderRail(stages){
  const byName = {};
  let max = 0, total = 0;
  (stages || []).forEach((s) => {
    byName[s.stage] = s;
    max = Math.max(max, s.duration_ms || 0);
    total += s.duration_ms || 0;
  });
  $("rail").innerHTML = STAGES.map(([key, label]) => {
    const s = byName[key];
    const cls = !s ? "" : (s.status === "failed" ? "failed" : "done");
    const width = s && max ? Math.max(2, (s.duration_ms / max) * 100) : 0;
    return '<li class="' + cls + '"><div class="name">' + esc(label) + "</div>" +
      '<div class="ms">' + (s ? num(s.duration_ms, 1) + " ms" : "&mdash;") + "</div>" +
      '<div class="bar"><i style="width:' + width + '%"></i></div></li>';
  }).join("");
  $("railtotal").textContent = stages && stages.length
    ? "total " + num(total, 1) + " ms sur " + stages.length + " phases"
    : "en attente";
}

/* ---------- ask ---------- */
async function ask(){
  const question = $("q").value.trim();
  if (!question) return;
  $("send").disabled = true;
  $("send").innerHTML = '<span class="spin"></span>';
  $("banners").innerHTML = "";
  $("answer").className = "answer empty";
  $("answer").textContent = "Inference en cours sur cette machine\u2026";
  $("sources").innerHTML = "";
  $("verdict").innerHTML = "";
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question})
    });
    render(await res.json());
  } catch (e) {
    $("answer").className = "answer";
    $("answer").textContent = "Echec de la requete : " + e;
  } finally {
    $("send").disabled = false;
    $("send").textContent = "Interroger";
  }
}
$("send").onclick = ask;
$("q").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") ask();
});

function render(data){
  const banners = [];
  if (data.error){
    banners.push('<div class="banner bad"><b>' + esc(data.error.code) + "</b> " +
      esc(data.error.message) + (data.error.recommended_action
        ? "<br>&rarr; " + esc(data.error.recommended_action) : "") + "</div>");
  }
  if (data.simulated){
    banners.push('<div class="banner">Backend <b>simule</b> : ce texte est extrait ' +
      "du corpus, aucun modele ne l'a genere. Chargez un fichier GGUF pour des " +
      "resultats exploitables.</div>");
  }
  (data.dic_violations || []).forEach((v) => {
    banners.push('<div class="banner bad">Contrat deterministe <b>' + esc(v.rule) +
      "</b> : " + esc(v.detail) + "</div>");
  });
  $("banners").innerHTML = banners.join("");

  $("answer").className = "answer";
  $("answer").textContent = data.answer || "(aucune reponse)";

  $("sources").innerHTML = (data.evidence || []).map((e) =>
    '<li><span class="label">' + esc(e.label) + "</span>" +
    "<span>" + esc(e.document_name) + '<div class="where">' +
    esc((e.section_path || []).join(" \u203a ") || "\u2014") + "</div></span>" +
    '<span class="score">' + num(e.score, 3) + "</span>" +
    '<div class="excerpt">' + esc(e.excerpt || "") + "</div></li>"
  ).join("");

  const v = data.validation;
  if (v){
    const cls = v.passed ? "ok" : (v.confidence === "insufficient" ? "warn" : "bad");
    const badges = [
      '<span class="badge ' + cls + '">' + (v.passed ? "valide" : "non valide") +
        " &middot; confiance <b>" + esc(v.confidence) + "</b></span>",
      '<span class="badge">ancrage <b>' + num(v.grounding_rate * 100, 0) + "%</b></span>",
      '<span class="badge">couverture des extraits <b>' +
        num(v.evidence_coverage * 100, 0) + "%</b></span>"
    ];
    (v.warnings || []).forEach((w) =>
      badges.push('<span class="badge warn">' + esc(w) + "</span>"));
    $("verdict").innerHTML = badges.join("");
  }

  renderRail(data.stages || []);
  renderProof(data.metrics || {}); renderMetrics(data.metrics || {});
  renderKV("budget", data.budget || {});
  renderKV("valdetail", v ? {
    "Phrases non etayees": (v.unsupported_sentences || []).join(" | ") || "aucune",
    "Etiquettes citees": (v.cited_labels || []).join(", ") || "aucune",
    "Insuffisance declaree": v.uncertainty_declared ? "oui" : "non",
    "Controles": Object.entries(v.checks || {})
      .map(([k, ok]) => (ok ? "+" : "-") + " " + k).join("  ")
  } : {});
  $("raw").textContent = JSON.stringify(data.iec || {}, null, 2);
  loadStatus();
}

function renderProof(m){
  // Act 5 of the demonstration: the six numbers a jury is asked to read, plus
  // the claim that carries the product. Model and quantization come from the
  // profile, so the panel says something true even before the first question.
  const p = (STATUS && STATUS.profile) || {}, b = (STATUS && STATUS.backend) || {};
  const model = (p.model && p.model.id) || "?";
  const quant = (p.model && p.model.quantization) || "?";
  const r = (STATUS && STATUS.resources) || {};
  m = m || {};
  const offlineOk = STATUS && STATUS.offline_enforced;
  const cells = [
    ["hors ligne", offlineOk ? "OUI" : "NON", offlineOk ? "aucun reseau" : "non applique",
     "offline" + (offlineOk ? "" : " bad")],
    ["modele", model, quant, ""],
    ["debit", b.is_simulated ? "\u2014" : num(m.tokens_per_second, 1),
     b.is_simulated ? "moteur simule" : "tokens/s", ""],
    ["latence", b.is_simulated ? "\u2014" : num(m.first_token_latency_ms, 0),
     b.is_simulated ? "moteur simule" : "ms au 1er token", ""],
    ["memoire", num(m.peak_rss_mb || r.process_rss_mb, 0), "Mo au pic", ""],
    ["temperature", m.temperature_c == null && r.temperature_c == null
      ? "n/d" : num(m.temperature_c != null ? m.temperature_c : r.temperature_c, 0),
     (r.thermal_source && r.thermal_source !== "unavailable") ? "\u00b0C" : "aucun capteur", ""]
  ];
  $("proof").innerHTML = cells.map(([k, v, u, cls]) =>
    '<div class="cell ' + cls + '"><div class="k">' + esc(k) + '</div><div class="v">' +
    esc(String(v)) + '</div><div class="u">' + esc(u) + "</div></div>").join("");
}

function renderMetrics(m){
  const cells = [
    ["debit", m.tokens_per_second, "tok/s", 2],
    ["1er token", m.first_token_latency_ms, "ms", 0],
    ["inference", m.inference_ms, "ms", 0],
    ["total", m.total_ms, "ms", 0],
    ["prompt", m.prompt_tokens, "tokens", 0],
    ["generes", m.generated_tokens, "tokens", 0],
    ["RSS max", m.peak_rss_mb, "Mo", 0],
    ["RAM libre", m.available_ram_mb, "Mo", 0],
    ["temperature", m.temperature_c, "\u00b0C", 1],
    ["compression", m.context_compression_ratio, "corpus \u2192 prompt", 3],
    ["precision recherche", m.retrieval_precision, "", 2],
    ["efficacite prompt", m.prompt_efficiency, "preuves/prompt", 2],
    ["ancrage", m.response_grounding_rate, "", 2]
  ];
  $("metrics").innerHTML = cells.map(([k, v, u, d]) =>
    '<div class="cell' + (v === null || v === undefined ? " muted" : "") + '">' +
    '<div class="k">' + esc(k) + '</div><div class="v">' + num(v, d) +
    '</div><div class="u">' + esc(u) + "</div></div>"
  ).join("");
}

function renderKV(id, obj){
  const rows = Object.entries(obj || {});
  $(id).innerHTML = rows.length
    ? rows.map(([k, v]) => "<tr><td>" + esc(k) + "</td><td>" + esc(v) + "</td></tr>").join("")
    : '<tr><td colspan="2" class="hint">aucune donnee</td></tr>';
}

loadStatus();
</script>
</body>
</html>
"""
