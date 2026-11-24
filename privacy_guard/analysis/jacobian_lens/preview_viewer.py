# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Self-contained interactive HTML viewer for ``privacyguard.jlens_preview/v1``.

The page inlines all data, CSS and JavaScript and sets a Content-Security-Policy
that forbids network requests, so it works offline and makes no model, API or
CDN calls. A URL fragment such as ``#r=0&method=jlens&color=spider&word=1&layer=26&pos=11``
preselects the view.
"""

import hashlib
import html
import json
import os
import re
from collections.abc import Sequence
from typing import Any

from privacy_guard.analysis.jacobian_lens.preview_analysis_input import (
    JacobianLensPreviewInput,
)
from privacy_guard.analysis.jacobian_lens.preview_analysis_node import (
    HEAD_SCORE_TOLERANCE,
    JacobianLensPreviewNode,
)
from privacy_guard.analysis.jacobian_lens.preview_schema import (
    canonical_json,
    JLensPreviewReceipt,
)


DEFAULT_TITLE = "PrivacyGuard J-Lens preview"
DEFAULT_CONTACT_LINE = "For questions, reach out to Matthew Grange"
HTML_FILENAME = "jlens_preview.html"
RECEIPTS_FILENAME = "jlens_preview_receipts.json"
SUMMARIES_FILENAME = "jlens_preview_summaries.json"
CHECKSUMS_FILENAME = "SHA256SUMS"
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'"
)

_PLACEHOLDER_PATTERN = "__CSP__|__TITLE__|__CONTACT__|__RECEIPTS__|__SUMMARIES__"

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="__CSP__">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1c2b33;background:#fff}
h1{margin:0;font-size:24px}
.subtitle{margin:4px 0 16px;color:#5d6c7b}
h2{font-size:17px;margin:22px 0 8px}
.banner{padding:8px 12px;border-radius:6px;margin:6px 0;font-weight:600}
.banner.fixture{background:#fde7e9;color:#a0001c;border:1px solid #f0a0aa}
.banner.control{background:#fff4d6;color:#7a5200;border:1px solid #f0d28a}
.banner.target{background:#e6f2ff;color:#00366d;border:1px solid #9cc7f5}
.label{background:#f5f6f7;border-left:3px solid #8a96a3;padding:4px 8px;margin:3px 0;font-size:13px}
.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:10px 0}
.controls label{font-size:14px}
table{border-collapse:collapse;font-size:13px}
td,th{border:1px solid #dde1e5;padding:3px 6px;text-align:left;vertical-align:top}
th{background:#f5f6f7}
table.grid td.cell{cursor:pointer;min-width:58px;max-width:90px;overflow:hidden;white-space:nowrap;text-align:center;font-family:ui-monospace,Menlo,monospace;font-size:12px}
table.grid td.cell.sel{outline:3px solid #ff7a00;outline-offset:-3px}
table.grid td.na{background:repeating-linear-gradient(45deg,#eee,#eee 4px,#fafafa 4px,#fafafa 8px);color:#888}
.mono{font-family:ui-monospace,Menlo,monospace}
.cols{display:flex;flex-wrap:wrap;gap:18px}
.muted{color:#6b7885;font-size:12px}
pre{background:#f5f6f7;padding:8px;overflow:auto;max-height:360px;font-size:12px}
.legend span{display:inline-block;margin-right:12px;font-size:12px}
.scroll{overflow-x:auto;max-width:100%}
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="subtitle">__CONTACT__</div>
<div id="app"></div>
<script type="application/json" id="jlens-receipts">__RECEIPTS__</script>
<script type="application/json" id="jlens-summaries">__SUMMARIES__</script>
<script>
(function () {
  "use strict";
  var RECEIPTS = JSON.parse(document.getElementById("jlens-receipts").textContent);
  var SUMMARIES = JSON.parse(document.getElementById("jlens-summaries").textContent);
  var WORD_RE = /^[ \\u2581\\u0120]?(?:[A-Za-z]+|[0-9]+)$/;
  var CONCEPT_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"];
  var METHOD_NAMES = {jlens: "J-lens", logit_lens: "Logit lens"};
  var state = {r: 0, method: null, color: "top1", word: false, layer: null, pos: null};
  var app = document.getElementById("app");

  function el(tag, attrs) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      var v = attrs[k];
      if (k === "class") { e.className = v; }
      else if (k.indexOf("on") === 0) { e.addEventListener(k.slice(2), v); }
      else { e.setAttribute(k, v); }
    });
    for (var i = 2; i < arguments.length; i++) {
      var kid = arguments[i];
      if (kid === null || kid === undefined) { continue; }
      e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return e;
  }
  function svgEl(tag, attrs) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    return e;
  }
  function vis(t) {
    var s = t.replace(/ /g, "\\u2423").replace(/\\n/g, "\\u21b5").replace(/\\t/g, "\\u21e5").replace(/\\r/g, "\\u240d");
    return s.length ? s : "\\u2205";
  }
  function R() { return RECEIPTS[state.r]; }
  function S() { return SUMMARIES[state.r]; }
  function cellIndex() {
    var idx = {};
    R().cells.forEach(function (c) { idx[c.method + "|" + c.layer + "|" + c.position] = c; });
    return idx;
  }
  function conceptOf() {
    var m = {};
    R().tracked_tokens.forEach(function (t) { m[t.token_id] = t.concept; });
    return m;
  }
  function concepts() {
    var seen = {};
    R().tracked_tokens.forEach(function (t) { seen[t.concept] = true; });
    return Object.keys(seen).sort();
  }
  function layers() {
    var seen = {};
    R().cells.forEach(function (c) { seen[c.layer] = true; });
    return Object.keys(seen).map(Number).sort(function (a, b) { return a - b; });
  }
  function bestRank(tracked, concept) {
    var cof = conceptOf(), best = null;
    tracked.forEach(function (t) {
      if (cof[t.token_id] === concept && (best === null || t.rank < best.rank)) { best = t; }
    });
    return best;
  }
  function shown(list) {
    return state.word ? list.filter(function (t) { return WORD_RE.test(t.text); }) : list;
  }
  function shade(rank) {
    var f = 1 - Math.log10(Math.max(rank, 1)) / Math.log10(Math.max(R().vocab_size, 10));
    return Math.round(97 - Math.max(0, Math.min(1, f)) * 57);
  }

  function banner() {
    var r = R(), div = el("div");
    if (r.model_role === "fixture") {
      div.append(el("div", {class: "banner fixture"}, "FIXTURE (" + r.run_kind + "): not a real-model result and not a Glimmer result"));
    } else if (r.model_role === "independent_control") {
      div.append(el("div", {class: "banner control"}, "Independent control model " + r.model_identity.repo_id + ": not a substitute for the Glimmer target"));
    } else {
      div.append(el("div", {class: "banner target"}, "Target model " + r.model_identity.repo_id + " @ " + r.model_identity.revision_sha + " (" + r.model_identity.dtype + ")"));
    }
    S().labels.forEach(function (l) { div.append(el("div", {class: "label"}, l)); });
    return div;
  }

  function controls() {
    var div = el("div", {class: "controls"});
    if (RECEIPTS.length > 1) {
      var rs = el("select", {onchange: function (e) { state.r = Number(e.target.value); reset(); render(); }});
      RECEIPTS.forEach(function (r, i) {
        var o = el("option", {value: String(i)}, (r.prompt.label || "prompt " + (i + 1)) + " | " + r.model_identity.repo_id);
        if (i === state.r) { o.selected = true; }
        rs.append(o);
      });
      div.append(el("label", {}, "Receipt ", rs));
    }
    var ms = el("select", {onchange: function (e) { state.method = e.target.value; render(); }});
    ["jlens", "logit_lens"].forEach(function (m) {
      var o = el("option", {value: m}, METHOD_NAMES[m]);
      if (m === state.method) { o.selected = true; }
      ms.append(o);
    });
    div.append(el("label", {}, "Grid method ", ms));
    var cs = el("select", {onchange: function (e) { state.color = e.target.value; render(); }});
    ["top1"].concat(concepts()).forEach(function (c) {
      var o = el("option", {value: c}, c === "top1" ? "top-1 score" : "rank of " + c);
      if (c === state.color) { o.selected = true; }
      cs.append(o);
    });
    div.append(el("label", {}, "Color by ", cs));
    var wb = el("input", {type: "checkbox", onchange: function (e) { state.word = e.target.checked; render(); }});
    wb.checked = state.word;
    div.append(el("label", {}, wb, " Display filter: word-like tokens only (letters or digits, optional leading space). Ranks shown stay unfiltered full-vocabulary ranks."));
    return div;
  }

  function promptRow() {
    var r = R(), t = el("table");
    var head = el("tr", {}, el("th", {}, "position"));
    var ids = el("tr", {}, el("th", {}, "token id"));
    r.prompt.token_texts.forEach(function (tx, i) {
      head.append(el("th", {class: "mono"}, i + ": " + vis(tx)));
      ids.append(el("td", {class: "mono"}, r.prompt.token_ids[i]));
    });
    t.append(head, ids);
    return el("div", {},
      el("div", {class: "mono"}, JSON.stringify(r.prompt.text)),
      el("div", {class: "muted"}, "rendering: " + r.prompt.rendering + " | BOS: " + (r.prompt.bos_token_id === null ? "none prepended" : "id " + r.prompt.bos_token_id) + (r.prompt.note ? " | " + r.prompt.note : "") + " | sha256 " + r.prompt.sha256),
      t);
  }

  function grid() {
    var r = R(), idx = cellIndex(), t = el("table", {class: "grid"});
    var scores = r.cells.filter(function (c) { return c.method === state.method && c.status === "ok"; }).map(function (c) { return c.top[0].score; });
    var lo = scores.length ? Math.min.apply(null, scores) : 0, hi = scores.length ? Math.max.apply(null, scores) : 1;
    var head = el("tr", {}, el("th", {}, "layer \\\\ position"));
    r.prompt.token_texts.forEach(function (tx, i) { head.append(el("th", {class: "mono"}, i + ": " + vis(tx))); });
    t.append(head);
    layers().slice().reverse().forEach(function (layer) {
      var tr = el("tr", {}, el("th", {}, layer));
      r.prompt.token_ids.forEach(function (_, pos) {
        var c = idx[state.method + "|" + layer + "|" + pos];
        var sel = layer === state.layer && pos === state.pos ? " sel" : "";
        var click = function () { state.layer = layer; state.pos = pos; render(); };
        if (!c || c.status !== "ok") {
          tr.append(el("td", {class: "cell na" + sel, title: c ? c.unavailable_reason : "not computed", onclick: click}, "n/a"));
          return;
        }
        var top = shown(c.top), label = top.length ? vis(top[0].text) + (state.word && top[0].rank > 1 ? " #" + top[0].rank : "") : "(none in top-" + c.top.length + ")";
        var light;
        if (state.color === "top1") {
          light = Math.round(97 - (hi > lo ? (c.top[0].score - lo) / (hi - lo) : 1) * 57);
        } else {
          var b = bestRank(c.tracked, state.color);
          light = b ? shade(b.rank) : 97;
        }
        var style = "background:hsl(212,70%," + light + "%);color:" + (light < 58 ? "#fff" : "#1c2b33");
        var tip = state.color === "top1" ? "top-1 score " + c.top[0].score.toFixed(3) : state.color + " best rank " + (bestRank(c.tracked, state.color) || {rank: "?"}).rank;
        tr.append(el("td", {class: "cell" + sel, style: style, title: tip, onclick: click}, label));
      });
      t.append(tr);
    });
    return t;
  }

  function rankTable(c, title) {
    var box = el("div", {}, el("h3", {}, title));
    if (!c) { box.append(el("div", {class: "muted"}, "not computed")); return box; }
    if (c.status !== "ok") { box.append(el("div", {class: "muted"}, "unavailable: " + c.unavailable_reason)); return box; }
    var t = el("table", {}, el("tr", {}, el("th", {}, "rank"), el("th", {}, "token"), el("th", {}, "raw text (JSON)"), el("th", {}, "id"), el("th", {}, "score"), el("th", {}, "ties")));
    shown(c.top).forEach(function (e) {
      t.append(el("tr", {}, el("td", {}, e.rank), el("td", {class: "mono"}, vis(e.text)), el("td", {class: "mono"}, JSON.stringify(e.text)), el("td", {class: "mono"}, e.token_id), el("td", {}, e.score.toFixed(4)), el("td", {}, e.ties)));
    });
    box.append(t, el("div", {class: "muted"}, state.word ? "Display filter on: " + shown(c.top).length + " of " + c.top.length + " top entries shown; ranks are unfiltered." : "Unfiltered top-" + c.top.length + " of " + R().vocab_size + "."));
    return box;
  }

  function trackedTable(cells) {
    var cof = conceptOf();
    var t = el("table", {}, el("tr", {}, el("th", {}, "concept"), el("th", {}, "token"), el("th", {}, "id"),
      el("th", {}, "J-lens rank"), el("th", {}, "logit-lens rank"), el("th", {}, "actual next-token rank")));
    var actual = {};
    if (R().actual_output) { R().actual_output.next_token_tracked.forEach(function (e) { actual[e.token_id] = e.rank; }); }
    R().tracked_tokens.forEach(function (tok) {
      var ranks = ["jlens", "logit_lens"].map(function (m) {
        var c = cells[m];
        if (!c || c.status !== "ok") { return "n/a"; }
        var e = c.tracked.filter(function (x) { return x.token_id === tok.token_id; })[0];
        return e ? e.rank : "n/a";
      });
      t.append(el("tr", {}, el("td", {}, cof[tok.token_id]), el("td", {class: "mono"}, vis(tok.text)), el("td", {class: "mono"}, tok.token_id),
        el("td", {}, ranks[0]), el("td", {}, ranks[1]), el("td", {}, actual[tok.token_id] === undefined ? "n/a" : actual[tok.token_id])));
    });
    return t;
  }

  function detail() {
    var idx = cellIndex(), r = R();
    var cells = {jlens: idx["jlens|" + state.layer + "|" + state.pos], logit_lens: idx["logit_lens|" + state.layer + "|" + state.pos]};
    var tok = r.prompt.token_texts[state.pos];
    var box = el("div", {}, el("h2", {}, "Cell: layer " + state.layer + ", position " + state.pos + " (" + vis(tok === undefined ? "" : tok) + ", id " + r.prompt.token_ids[state.pos] + ")"),
      el("div", {class: "muted"}, "scores: " + r.score_definition));
    box.append(el("div", {class: "cols"}, rankTable(cells.jlens, "J-lens"), rankTable(cells.logit_lens, "Logit lens"), actualBox()));
    box.append(el("h3", {}, "Tracked tokens at this cell"), trackedTable(cells));
    return box;
  }

  function actualBox() {
    var a = R().actual_output, box = el("div", {}, el("h3", {}, "Actual model output"));
    if (!a) { box.append(el("div", {class: "muted"}, "unavailable")); return box; }
    box.append(el("div", {class: "mono"}, "greedy (" + a.max_new_tokens + " max new tokens): " + JSON.stringify(a.greedy_text)));
    box.append(el("div", {class: "muted"}, "greedy ids: " + a.greedy_ids.join(", ")));
    var t = el("table", {}, el("tr", {}, el("th", {}, "rank"), el("th", {}, "next token"), el("th", {}, "id"), el("th", {}, "score")));
    shown(a.next_token_top).forEach(function (e) {
      t.append(el("tr", {}, el("td", {}, e.rank), el("td", {class: "mono"}, vis(e.text)), el("td", {class: "mono"}, e.token_id), el("td", {}, e.score.toFixed(4))));
    });
    box.append(t);
    var h = S().head_consistency;
    box.append(el("div", {class: "muted"}, h.checked ? "final-layer logit lens vs actual: top-1 match " + h.top1_match + ", max |score diff| " + (h.max_abs_score_diff === null ? "n/a (no shared top entries)" : h.max_abs_score_diff) + " (tolerance " + h.tolerance + ")" : "head consistency not checked: " + h.reason));
    return box;
  }

  function curves(curveList, title) {
    var box = el("div", {}, el("h2", {}, title));
    var list = curveList.filter(function (c) { return c.position === state.pos; });
    if (!list.length) { box.append(el("div", {class: "muted"}, "no cells at this position")); return box; }
    var cs = concepts(), W = 680, H = 260, P = 44, V = Math.max(R().vocab_size, 10);
    var allLayers = layers(), lmin = allLayers[0], lmax = allLayers[allLayers.length - 1];
    var x = function (l) { return P + (lmax > lmin ? (l - lmin) / (lmax - lmin) : 0.5) * (W - 2 * P); };
    var y = function (rank) { return P / 2 + Math.log10(rank) / Math.log10(V) * (H - P); };
    var svg = svgEl("svg", {width: W, height: H + 10, role: "img"});
    [1, 10, 100, 1000, 10000, 100000].filter(function (v) { return v <= V; }).forEach(function (v) {
      svg.append(svgEl("line", {x1: P, x2: W - P, y1: y(v), y2: y(v), stroke: "#e4e7ea"}));
      var t = svgEl("text", {x: 4, y: y(v) + 4, "font-size": 10, fill: "#6b7885"}); t.textContent = "#" + v; svg.append(t);
    });
    var step = Math.max(1, Math.ceil((lmax - lmin + 1) / 13));
    allLayers.filter(function (l) { return (l - lmin) % step === 0 || l === lmax; }).forEach(function (l) {
      var t = svgEl("text", {x: x(l) - 4, y: H + 8, "font-size": 10, fill: "#6b7885"}); t.textContent = String(l); svg.append(t);
    });
    var xl = svgEl("text", {x: W - P + 6, y: H + 8, "font-size": 10, fill: "#6b7885"}); xl.textContent = "layer"; svg.append(xl);
    var legend = el("div", {class: "legend"});
    list.forEach(function (c) {
      var color = CONCEPT_COLORS[cs.indexOf(c.concept) % CONCEPT_COLORS.length];
      var pts = [];
      c.layers.forEach(function (l, i) { if (c.best_rank[i] !== null) { pts.push(x(l) + "," + y(c.best_rank[i])); } });
      if (pts.length) {
        svg.append(svgEl("polyline", {points: pts.join(" "), fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": c.method === "jlens" ? "" : "5,4"}));
      }
      c.layers.forEach(function (l, i) {
        if (c.best_rank[i] === null) { return; }
        var dot = svgEl("circle", {cx: x(l), cy: y(c.best_rank[i]), r: 3, fill: c.method === "jlens" ? color : "#fff", stroke: color});
        var tip = svgEl("title", {}); tip.textContent = c.concept + " " + METHOD_NAMES[c.method] + " layer " + l + ": rank " + c.best_rank[i];
        dot.append(tip); svg.append(dot);
      });
      legend.append(el("span", {style: "color:" + color}, c.concept + " (" + METHOD_NAMES[c.method] + (c.method === "jlens" ? ", solid" : ", dashed") + ")"));
    });
    box.append(svg, legend, el("div", {class: "muted"}, "Best full-vocabulary rank over each concept's tracked token variants (log scale, rank 1 at top); gaps are unavailable cells."));
    return box;
  }

  function compactRange(xs) {
    if (!xs.length) { return ""; }
    var out = [], start = xs[0], prev = xs[0];
    xs.slice(1).concat([null]).forEach(function (x) {
      if (x !== null && x === prev + 1) { prev = x; return; }
      out.push(start === prev ? String(start) : start + "-" + prev);
      start = prev = x;
    });
    return out.join(",");
  }

  function interventions() {
    var r = R(), box = el("div", {}, el("h2", {}, "Interventions and controls"));
    if (!r.interventions.length) { box.append(el("div", {class: "muted"}, "none in this receipt")); return box; }
    var margins = {};
    S().answer_margins.forEach(function (m) { margins[m.name] = m; });
    var head = el("tr");
    ["name", "kind", "primary", "method", "layers", "positions", "strength", "delta norm", "direction", "next-token top-5", "answer margin", "greedy text", "status"].forEach(function (h) { head.append(el("th", {}, h)); });
    var t = el("table", {}, head);
    r.interventions.forEach(function (run) {
      var m = margins[run.name];
      t.append(el("tr", {},
        el("td", {}, run.name), el("td", {}, run.kind), el("td", {}, run.primary ? "primary (predeclared)" : ""), el("td", {}, run.method_label),
        el("td", {}, compactRange(run.layers)), el("td", {}, compactRange(run.positions)),
        el("td", {}, run.strength === null ? "" : run.strength), el("td", {}, run.delta_norm === null ? "" : run.delta_norm),
        el("td", {class: "mono"}, run.direction ? JSON.stringify(run.direction) : ""),
        el("td", {class: "mono"}, run.next_token_top.slice(0, 5).map(function (e) { return vis(e.text) + "#" + e.rank; }).join(" ")),
        el("td", {}, m ? m.preferred_concept + " - " + m.reference_concept + " = " + m.margin.toFixed(4) : ""),
        el("td", {class: "mono"}, run.greedy_text === null ? "" : JSON.stringify(run.greedy_text)),
        el("td", {}, run.status + (run.unavailable_reason ? ": " + run.unavailable_reason : ""))));
    });
    box.append(el("div", {class: "scroll"}, t));
    S().intervention_readouts.forEach(function (ir) {
      box.append(el("h3", {}, "Representation readout after " + ir.name + " (" + ir.kind + ")"));
      var rt = el("table", {}, el("tr", {}, el("th", {}, "method"), el("th", {}, "layer"), el("th", {}, "position"), el("th", {}, "concept"), el("th", {}, "unpatched rank"), el("th", {}, "patched rank")));
      ir.rank_changes.forEach(function (c) {
        rt.append(el("tr", {}, el("td", {}, METHOD_NAMES[c.method]), el("td", {}, c.layer), el("td", {}, c.position), el("td", {}, c.concept),
          el("td", {}, c.baseline_rank === null ? "n/a" : c.baseline_rank), el("td", {}, c.patched_rank)));
      });
      box.append(rt, curves(ir.curves, "Rank curves of the patched pass"));
    });
    return box;
  }

  function provenance() {
    var r = R(), box = el("div", {}, el("h2", {}, "Provenance"));
    var prov = {schema_version: r.schema_version, receipt_sha256: r.receipt_sha256, content_sha256: S().receipt_sha256, run_kind: r.run_kind,
      model_role: r.model_role, created_utc: r.created_utc, model_identity: r.model_identity, code: r.code, lens: r.lens,
      score_definition: r.score_definition, num_layers: r.num_layers, vocab_size: r.vocab_size, top_k: r.top_k,
      numerical_receipt: r.numerical_receipt, display: r.display, unavailable: r.unavailable, cell_counts: S().num_cells};
    box.append(el("pre", {}, JSON.stringify(prov, null, 2)));
    var a = el("a", {download: "jlens_preview_receipt_" + (state.r + 1) + ".json",
      href: "data:application/json;charset=utf-8," + encodeURIComponent(JSON.stringify(r, null, 2))}, "Download this receipt (raw JSON)");
    box.append(a);
    return box;
  }

  function reset() {
    var r = R();
    state.method = r.cells.some(function (c) { return c.method === "jlens" && c.status === "ok"; }) ? "jlens" : "logit_lens";
    var ls = layers();
    state.layer = ls.length ? ls[ls.length - 1] : 0;
    state.pos = r.prompt.token_ids.length - 1;
  }

  function render() {
    app.replaceChildren(banner(), controls(), el("h2", {}, "Prompt"), promptRow(),
      el("h2", {}, METHOD_NAMES[state.method] + " readout: layer x token position (click a cell)"), el("div", {class: "scroll"}, grid()),
      detail(), curves(S().rank_curves, "Rank curves at position " + state.pos), interventions(), provenance());
  }

  function applyHash() {
    var params = {};
    location.hash.replace(/^#/, "").split("&").forEach(function (kv) {
      var i = kv.indexOf("=");
      if (i <= 0) { return; }
      try { params[decodeURIComponent(kv.slice(0, i))] = decodeURIComponent(kv.slice(i + 1)); } catch (e) { /* ignore malformed fragment */ }
    });
    if (params.r !== undefined && RECEIPTS[Number(params.r)]) { state.r = Number(params.r); }
    reset();
    if (params.method === "jlens" || params.method === "logit_lens") { state.method = params.method; }
    if (params.color) { state.color = params.color; }
    if (params.word === "1") { state.word = true; }
    if (params.layer !== undefined && isFinite(Number(params.layer))) { state.layer = Number(params.layer); }
    if (params.pos !== undefined && isFinite(Number(params.pos))) { state.pos = Number(params.pos); }
  }

  applyHash();
  render();
})();
</script>
</body>
</html>
"""


def _script_json(data: Any) -> str:
    return (
        json.dumps(data, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def summarize_receipt(
    receipt: JLensPreviewReceipt, head_score_tolerance: float = HEAD_SCORE_TOLERANCE
) -> dict[str, Any]:
    """Runs ``JacobianLensPreviewNode`` and returns its dict output."""
    node = JacobianLensPreviewNode(
        JacobianLensPreviewInput(receipt), head_score_tolerance=head_score_tolerance
    )
    return node.compute_outputs()


def render_preview_html(
    receipts: Sequence[JLensPreviewReceipt],
    summaries: Sequence[dict[str, Any]] | None = None,
    title: str = DEFAULT_TITLE,
    contact_line: str = DEFAULT_CONTACT_LINE,
) -> str:
    """
    Render one self-contained HTML page for one or more receipts.

    args:
        receipts: validated receipts, e.g. one per predeclared prompt
        summaries: node outputs aligned with receipts; computed if omitted
        title: page H1
        contact_line: subtitle shown directly under the H1
    """
    if not receipts:
        raise ValueError("At least one receipt is required")
    summary_list = (
        [summarize_receipt(r) for r in receipts]
        if summaries is None
        else list(summaries)
    )
    if len(summary_list) != len(receipts):
        raise ValueError("summaries must align with receipts")
    values = {
        "__CSP__": html.escape(CONTENT_SECURITY_POLICY, quote=False),
        "__TITLE__": html.escape(title),
        "__CONTACT__": html.escape(contact_line),
        "__RECEIPTS__": _script_json([r.to_dict() for r in receipts]),
        "__SUMMARIES__": _script_json(summary_list),
    }
    # One pass, so placeholder-like text inside receipts is never substituted.
    return re.sub(_PLACEHOLDER_PATTERN, lambda m: values[m.group(0)], _PAGE)


def write_preview_bundle(
    receipts: Sequence[JLensPreviewReceipt],
    out_dir: str,
    title: str = DEFAULT_TITLE,
    contact_line: str = DEFAULT_CONTACT_LINE,
    head_score_tolerance: float = HEAD_SCORE_TOLERANCE,
) -> dict[str, str]:
    """
    Write the HTML page, receipts JSON, node summaries JSON and SHA256SUMS.

    args:
        head_score_tolerance: passed to ``JacobianLensPreviewNode``; widen it only
            when the receipt's actual output and logit lens use different precisions

    returns:
        file name -> SHA-256 of each written file (excluding SHA256SUMS)
    """
    summaries = [summarize_receipt(r, head_score_tolerance) for r in receipts]
    contents = {
        HTML_FILENAME: render_preview_html(receipts, summaries, title, contact_line),
        RECEIPTS_FILENAME: canonical_json([r.to_dict() for r in receipts]),
        SUMMARIES_FILENAME: canonical_json(summaries),
    }
    os.makedirs(out_dir, exist_ok=True)
    digests = {}
    for name, text in contents.items():
        data = text.encode("utf-8")
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    with open(os.path.join(out_dir, CHECKSUMS_FILENAME), "w", encoding="utf-8") as f:
        f.writelines(f"{digest}  {name}\n" for name, digest in sorted(digests.items()))
    return digests
