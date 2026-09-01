"""Self-contained HTML and JSON report generation."""

from __future__ import annotations

import html
import csv
import json
from pathlib import Path
from typing import Any


PHASE_COLORS = {
    "Localization": "#8b5cf6",
    "Debugging": "#eab308",
    "Planning": "#3b82f6",
    "Patching": "#ef4444",
    "Validation": "#22c55e",
    "Recovery": "#ec4899",
    "General": "#64748b",
}


def _h(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _embed_json(value: Any) -> str:
    """Serialize JSON for safe embedding inside a <script> tag."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _page(title: str, body: str, data_json: str = "") -> str:
    data_block = (
        f'<script id="report-data" type="application/json">{data_json}</script>'
        if data_json
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_h(title)}</title>
<style>
:root {{ color-scheme: light dark; --bg:#f4f7fb; --panel:#fff; --text:#172033; --muted:#64748b; --line:#dce3ed; --accent:#2563eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1422; --panel:#151f31; --text:#e5edf8; --muted:#9dafc7; --line:#2b3a50; --accent:#60a5fa; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,-apple-system,sans-serif; }}
main {{ max-width:1320px; margin:auto; padding:24px; }}
h1,h2,h3,h4 {{ margin-top:0; }} h2 {{ margin-top:4px; }}
a {{ color:var(--accent); }}
.hero,.panel,.card,details {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; }}
.hero,.panel {{ padding:18px; margin-bottom:14px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }}
.card {{ padding:14px; min-width:0; }}
.label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
.value {{ font-size:18px; font-weight:700; overflow-wrap:anywhere; }}
.badge {{ display:inline-block; border-radius:999px; padding:3px 9px; font-size:12px; font-weight:700; background:#e2e8f0; color:#334155; }}
.answered {{ background:#dcfce7; color:#166534; }} .partial {{ background:#fef3c7; color:#92400e; }} .not_assessable {{ background:#e2e8f0; color:#475569; }}
.resolved {{ color:#15803d; }} .unresolved {{ color:#b91c1c; }}
.tabs {{ display:flex; gap:6px; flex-wrap:wrap; margin:6px 0 16px; }}
.tabs button {{ padding:8px 14px; border:1px solid var(--line); border-radius:999px; background:var(--panel); color:var(--text); cursor:pointer; font-weight:600; }}
.tabs button.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.tab-panel {{ display:none; }} .tab-panel.active {{ display:block; }}
input,button {{ padding:8px 11px; border:1px solid var(--line); border-radius:8px; background:var(--panel); color:var(--text); }}
textarea,select {{ padding:8px 11px; border:1px solid var(--line); border-radius:8px; background:var(--panel); color:var(--text); }}
textarea::placeholder,input::placeholder {{ color:var(--muted); }}
button {{ cursor:pointer; }}
details {{ margin:10px 0; overflow:hidden; }} summary {{ cursor:pointer; padding:12px 14px; font-weight:700; }}
.details-body {{ padding:0 14px 14px; }}
.evidence {{ border-left:3px solid var(--accent); padding:8px 12px; margin:8px 0; background:color-mix(in srgb,var(--panel) 82%,var(--accent)); border-radius:0 8px 8px 0; }}
blockquote {{ margin:4px 0; white-space:pre-wrap; overflow-wrap:anywhere; font-family:ui-monospace,monospace; font-size:13px; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; max-height:460px; overflow:auto; padding:12px; background:color-mix(in srgb,var(--panel) 82%,#64748b); border-radius:8px; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; vertical-align:top; border-bottom:1px solid var(--line); padding:9px; overflow-wrap:anywhere; }}
.muted {{ color:var(--muted); }} .hidden {{ display:none !important; }}

/* Graph explorer */
.explorer {{ display:grid; grid-template-columns:minmax(300px,440px) 1fr; gap:18px; align-items:start; }}
.flow {{ display:flex; flex-direction:column; align-items:center; padding:8px 4px; max-height:76vh; overflow:auto; }}
.fnode {{ position:relative; width:100%; max-width:360px; border:none; cursor:pointer; text-align:left; color:#fff; border-radius:12px; padding:12px 15px; background:var(--pc,#64748b); box-shadow:0 1px 5px rgba(0,0,0,.18); }}
.fnode:hover {{ filter:brightness(1.06); }}
.fnode.active {{ outline:3px solid var(--text); outline-offset:2px; }}
.fnode.context {{ background:var(--panel); color:var(--muted); border:1px dashed var(--line); box-shadow:none; }}
.fconn {{ width:2px; height:44px; background:var(--line); position:relative; }}
.fconn::after {{ content:"▼"; position:absolute; left:50%; bottom:-5px; transform:translateX(-50%); color:var(--muted); font-size:11px; }}
.fphase {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; opacity:.92; }}
.ftitle {{ font-weight:600; overflow-wrap:anywhere; margin-top:2px; }}
.fid {{ font-size:11px; opacity:.8; }}
.detail {{ position:sticky; top:14px; min-height:220px; }}
.detail .empty {{ color:var(--muted); padding:30px 6px; text-align:center; }}
.pill {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700; color:#fff; }}
.legend {{ display:flex; gap:12px; flex-wrap:wrap; margin:2px 0 14px; }}
.legend span {{ display:inline-flex; gap:6px; align-items:center; font-size:12px; color:var(--muted); }}
.legend i {{ width:12px; height:12px; border-radius:4px; display:inline-block; }}
/* Issue description */
.issue-text {{ white-space:pre-wrap; overflow-wrap:anywhere; font-family:ui-monospace,monospace; font-size:13px; line-height:2; padding:16px; background:color-mix(in srgb,var(--panel) 82%,#64748b); border-radius:10px; }}
mark.clue {{ padding:2px 5px; border-radius:5px; color:#0b1220; font-weight:600; box-shadow:inset 0 -2px 0 rgba(0,0,0,.18); cursor:help; }}
mark.cat-A {{ background:#c4b5fd; }} mark.cat-B {{ background:#7dd3fc; }}
mark.cat-C {{ background:#86efac; }} mark.cat-D {{ background:#fca5a5; }}
mark.cat-E {{ background:#fcd34d; }} mark.cat-F {{ background:#f0abfc; }}
mark.cat-G {{ background:#a5b4fc; }} mark.cat-H {{ background:#fdba74; }}
mark.cat-I {{ background:#94a3b8; }}
.clue-cat {{ display:inline-block; padding:2px 8px; border-radius:5px; font-size:12px; font-weight:700; color:#0b1220; }}
/* Raw trajectory */
.rawcard {{ border:1px solid var(--line); border-radius:12px; margin:12px 0; overflow:hidden; scroll-margin-top:80px; }}
.rawcard .raw-head {{ display:flex; gap:10px; align-items:center; padding:10px 14px; background:color-mix(in srgb,var(--panel) 80%,#64748b); font-weight:600; }}
.rawcard .raw-head .fid {{ margin-left:auto; }}
.rawtag {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.03em; padding:2px 8px; border-radius:999px; background:var(--accent); color:#fff; }}
.rawtag.system {{ background:#64748b; }} .rawtag.user {{ background:#0ea5e9; }}
.rawtag.assistant {{ background:#8b5cf6; }} .rawtag.tool {{ background:#f59e0b; }}
.raw-content {{ margin:0; border-radius:0; }}
.rawcard.flash {{ animation:flashcard 1.8s ease-out; }}
@keyframes flashcard {{ 0% {{ box-shadow:0 0 0 3px var(--accent); }} 100% {{ box-shadow:none; }} }}
mark.hit {{ background:#fde047; color:#0b1220; padding:1px 2px; border-radius:3px; }}
.evidence.linky {{ cursor:pointer; }} .evidence.linky:hover {{ filter:brightness(1.05); }}
.jumpbtn {{ margin-top:8px; }}
.inexact {{ margin-top:6px; font-size:12px; font-weight:600; color:#b45309; }}
@media (prefers-color-scheme: dark) {{ .inexact {{ color:#fbbf24; }} }}
.inexact-badge {{ display:inline-block; margin-top:6px; padding:2px 9px; border-radius:999px; font-size:11px; font-weight:700; background:#fef3c7; color:#92400e; }}
/* Manual evaluation form */
.evaltoolbar {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; position:sticky; top:0; z-index:3; padding:10px 0; margin-bottom:6px; background:var(--bg); }}
.evaltoolbar button.primary {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.evalstatus {{ color:var(--muted); font-size:13px; }}
.evalform .q {{ margin:14px 0; }}
.evalform label {{ display:block; font-weight:600; margin-bottom:6px; }}
.evalform textarea, .evalform select, .evalform input[type=text] {{ width:100%; font:inherit; }}
.evalform textarea {{ min-height:56px; resize:vertical; }}
.evalform select {{ max-width:260px; margin-bottom:8px; }}
@media (max-width:820px) {{ .explorer {{ grid-template-columns:1fr; }} .detail {{ position:static; }} }}
@media (max-width:650px) {{ main {{ padding:14px; }} }}
</style>
</head>
<body><main>{body}</main>
{data_block}
<script>
function switchTab(name){{
  document.querySelectorAll('.tab-panel').forEach(el=>el.classList.toggle('active',el.dataset.tab===name));
  document.querySelectorAll('.tabs button').forEach(el=>el.classList.toggle('active',el.dataset.tab===name));
}}
const dataEl=document.getElementById('report-data');
const REPORT=dataEl?JSON.parse(dataEl.textContent):null;
function el(tag,cls,text){{const n=document.createElement(tag);if(cls)n.className=cls;if(text!=null)n.textContent=text;return n;}}
function field(label,text){{const c=el('div','card');c.appendChild(el('div','label',label));const v=el('div');v.textContent=(text==null||text==='')?'—':String(text);c.appendChild(v);return c;}}
function jumpToEvent(eventId,quote){{
  switchTab('raw');
  const card=document.getElementById('raw-'+eventId);
  if(!card) return;
  const pre=card.querySelector('.raw-content');
  const ev=(REPORT.events||[]).find(e=>e.id===eventId);
  if(pre && ev){{
    pre.innerHTML='';
    const text=ev.content||'';
    const i=quote?text.indexOf(quote):-1;
    if(i>=0){{
      pre.appendChild(document.createTextNode(text.slice(0,i)));
      pre.appendChild(el('mark','hit',quote));
      pre.appendChild(document.createTextNode(text.slice(i+quote.length)));
    }} else {{
      pre.textContent=text;
    }}
  }}
  card.scrollIntoView({{behavior:'smooth',block:'center'}});
  card.classList.remove('flash'); void card.offsetWidth; card.classList.add('flash');
}}
function showNode(id){{
  if(!REPORT) return;
  document.querySelectorAll('.fnode').forEach(n=>n.classList.toggle('active',n.dataset.id===id));
  const node=(REPORT.nodes||[]).find(n=>n.id===id);
  const ctx=(REPORT.context||[]).find(n=>n.id===id);
  const panel=document.getElementById('detail');
  panel.innerHTML='';
  if(node){{
    const head=el('div');
    const pill=el('span','pill',node.phase);
    pill.style.background=node.color;
    head.appendChild(pill);
    head.appendChild(el('h3',null,node.title||node.id));
    head.appendChild(el('div','fid',node.id+' · events '+(node.event_ids||[]).join(', ')));
    if((node.evidence||[]).some(e=>e.exact_match===false)){{
      head.appendChild(el('div','inexact-badge','⚠ Contains approximate evidence'));
    }}
    panel.appendChild(head);
    if(node.summary) panel.appendChild(el('p',null,node.summary));
    const grid=el('div','grid');
    grid.appendChild(field('Intent',node.intent));
    grid.appendChild(field('Outcome',node.outcome));
    grid.appendChild(field('Resources',(node.resources||[]).join(', ')));
    grid.appendChild(field('Confidence',node.confidence));
    panel.appendChild(grid);
    panel.appendChild(el('h4',null,'Exact trajectory snippet'));
    (node.evidence||[]).forEach(ev=>{{
      const box=el('div','evidence linky');
      box.title='Open this snippet in the Raw trajectory tab';
      const lab=el('div','label',ev.event_id+' ↗');
      box.appendChild(lab);
      const q=el('blockquote');q.textContent=ev.quote||'';box.appendChild(q);
      if(ev.exact_match===false){{
        const w=el('div','inexact','⚠ Not an exact match — '+(ev.match_warning||'may differ from the source event.'));
        box.appendChild(w);
      }}
      if(ev.why) box.appendChild(el('div',null,ev.why));
      box.addEventListener('click',()=>jumpToEvent(ev.event_id,ev.quote));
      panel.appendChild(box);
    }});
    if(!(node.evidence||[]).length) panel.appendChild(el('p','muted','No direct quotation supplied.'));
  }} else if(ctx){{
    panel.appendChild(el('h3',null,ctx.title));
    panel.appendChild(el('div','fid',ctx.id+' · '+ctx.role+' · '+ctx.kind));
    panel.appendChild(el('p','muted','Context event (not part of the agent\\u2019s classified work).'));
    const pre=el('pre');pre.textContent=ctx.content||'';panel.appendChild(pre);
    const btn=el('button','jumpbtn','View in raw trajectory ↗');
    btn.addEventListener('click',()=>jumpToEvent(ctx.id,null));
    panel.appendChild(btn);
  }}
}}
document.addEventListener('DOMContentLoaded',()=>{{
  const first=document.querySelector('.fnode');
  if(first) showNode(first.dataset.id);
  initManualEval();
}});
function _evalForm(){{ return document.getElementById('eval-form'); }}
function _evalKey(){{ const f=_evalForm(); return f?('manual_eval::'+f.dataset.storageKey):null; }}
function _evalValues(){{
  const out={{}};
  document.querySelectorAll('[data-eval]').forEach(el=>{{ out[el.name]=el.value; }});
  return out;
}}
function _evalApply(values){{
  if(!values) return;
  document.querySelectorAll('[data-eval]').forEach(el=>{{ if(values[el.name]!=null) el.value=values[el.name]; }});
}}
function saveEval(){{
  const key=_evalKey(); if(!key) return;
  const payload={{case_id:_evalForm().dataset.storageKey, saved_at:new Date().toISOString(), values:_evalValues()}};
  try{{ localStorage.setItem(key,JSON.stringify(payload)); setEvalStatus('Saved to this browser · '+new Date().toLocaleString()); }}
  catch(e){{ setEvalStatus('Could not save: '+e.message); }}
}}
function initManualEval(){{
  const key=_evalKey(); if(!key) return;
  try{{ const raw=localStorage.getItem(key); if(raw){{ const p=JSON.parse(raw); _evalApply(p.values); setEvalStatus('Loaded saved evaluation · '+(p.saved_at?new Date(p.saved_at).toLocaleString():'')); }} }}
  catch(e){{}}
}}
function clearEval(){{
  const key=_evalKey(); if(!key) return;
  if(!confirm('Clear this manual evaluation from this browser?')) return;
  localStorage.removeItem(key);
  document.querySelectorAll('[data-eval]').forEach(el=>{{ el.value=''; }});
  setEvalStatus('Cleared.');
}}
function downloadEval(){{
  const f=_evalForm(); if(!f) return;
  const payload={{case_id:f.dataset.storageKey, saved_at:new Date().toISOString(), values:_evalValues()}};
  const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=f.dataset.storageKey+'-manual-eval.json';
  a.click(); URL.revokeObjectURL(a.href);
}}
function loadEvalFile(input){{
  const file=input.files&&input.files[0]; if(!file) return;
  const reader=new FileReader();
  reader.onload=()=>{{ try{{ const p=JSON.parse(reader.result); _evalApply(p.values||p); saveEval(); setEvalStatus('Loaded from file and saved.'); }} catch(e){{ setEvalStatus('Invalid file: '+e.message); }} }};
  reader.readAsText(file); input.value='';
}}
function setEvalStatus(text){{ const s=document.getElementById('eval-status'); if(s) s.textContent=text; }}
</script>
</body></html>"""


def _graph_entries(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the ordered clickable chain of analysis nodes and context events."""
    events = artifact.get("events", [])
    order = {event["id"]: index for index, event in enumerate(events)}
    event_map = {event["id"]: event for event in events}

    entries: list[tuple[int, dict[str, Any]]] = []
    for node in artifact.get("nodes", []):
        event_ids = node.get("event_ids", [])
        position = min((order.get(eid, 1_000_000) for eid in event_ids), default=1_000_000)
        phase = node.get("phase", "General")
        entries.append(
            (
                position,
                {
                    "kind": "node",
                    "id": node.get("id"),
                    "phase": phase,
                    "color": PHASE_COLORS.get(phase, PHASE_COLORS["General"]),
                    "title": node.get("title") or node.get("id"),
                    "summary": node.get("summary", ""),
                    "intent": node.get("intent", ""),
                    "outcome": node.get("outcome", ""),
                    "resources": node.get("resources", []),
                    "confidence": node.get("confidence"),
                    "event_ids": event_ids,
                    "evidence": node.get("evidence", []),
                },
            )
        )

    for event_id in artifact.get("unclassified_event_ids", []):
        event = event_map.get(event_id)
        if not event:
            continue
        text = (event.get("content") or "").strip().replace("\n", " ")
        entries.append(
            (
                order.get(event_id, 1_000_000),
                {
                    "kind": "context",
                    "id": event_id,
                    "phase": "Context",
                    "color": PHASE_COLORS["General"],
                    "title": (text[:70] + "…") if len(text) > 70 else (text or "(empty event)"),
                    "role": event.get("role"),
                    "kind_label": event.get("kind"),
                    "content": event.get("content", ""),
                },
            )
        )

    entries.sort(key=lambda item: item[0])
    return [entry for _, entry in entries]


def _render_graph(entries: list[dict[str, Any]]) -> str:
    legend = "".join(
        f'<span><i style="background:{color}"></i>{_h(phase)}</span>'
        for phase, color in PHASE_COLORS.items()
    )
    boxes = []
    for index, entry in enumerate(entries):
        if index:
            boxes.append('<div class="fconn"></div>')
        is_context = entry["kind"] == "context"
        cls = "fnode context" if is_context else "fnode"
        label = "Context" if is_context else entry["phase"]
        boxes.append(
            f'<button class="{cls}" data-id="{_h(entry["id"])}" '
            f'style="--pc:{entry["color"]}" onclick="showNode(\'{_h(entry["id"])}\')">'
            f'<div class="fphase">{_h(label)}</div>'
            f'<div class="ftitle">{_h(entry["title"])}</div>'
            f'<div class="fid">{_h(entry["id"])}</div>'
            "</button>"
        )
    if not boxes:
        boxes.append('<p class="muted">No nodes were produced for this trajectory.</p>')
    return (
        '<section class="panel"><h2>Trajectory graph</h2>'
        '<p class="muted">Click any node to open its details and the exact trajectory snippet.</p>'
        f'<div class="legend">{legend}</div>'
        '<div class="explorer">'
        f'<div class="flow">{"".join(boxes)}</div>'
        '<div class="detail card" id="detail"><div class="empty">Select a node to see details.</div></div>'
        "</div></section>"
    )


def _render_issue(artifact: dict[str, Any]) -> str:
    case = artifact["case"]
    issue = case.get("issue_description") or ""
    clues = sorted(
        (clue for clue in artifact["issue_clues"].get("clues", []) if isinstance(clue.get("start"), int)),
        key=lambda clue: (clue["start"], -(clue.get("end", 0))),
    )

    # Highlight non-overlapping clue spans over the raw issue text.
    highlighted = []
    cursor = 0
    for clue in clues:
        start, end = clue["start"], clue.get("end", clue["start"])
        if start < cursor or start >= len(issue):
            continue
        highlighted.append(_h(issue[cursor:start]))
        category = str(clue.get("category") or "")
        group = category[0] if category else "I"
        inexact = clue.get("exact_match") is False
        tip = f'{category} · {clue.get("type", "")}'
        if inexact:
            tip += " · approximate match"
        approx = " ≈" if inexact else ""
        highlighted.append(
            f'<mark class="clue cat-{_h(group)}" title="{_h(tip)}">{_h(issue[start:end])}{approx}</mark>'
        )
        cursor = end
    highlighted.append(_h(issue[cursor:]))
    issue_html = "".join(highlighted) or '<span class="muted">No issue description was found.</span>'

    rows = []
    for clue in artifact["issue_clues"].get("clues", []):
        category = str(clue.get("category") or "")
        group = category[0] if category else "I"
        warning = clue.get("match_warning")
        match_cell = (
            f'<span class="muted" title="{_h(warning)}">≈ approximate</span>'
            if warning
            else "exact"
        )
        rows.append(
            "<tr>"
            f'<td><span class="clue-cat cat-{_h(group)}">{_h(category)}</span></td>'
            f'<td>{_h(clue.get("type"))}</td>'
            f'<td>{_h(clue.get("value") or clue.get("quote"))}</td>'
            f'<td>{_h(clue.get("role"))}</td>'
            f'<td>{_h(clue.get("signal_strength"))}</td>'
            f'<td>{match_cell}</td>'
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="6" class="muted">No keywords extracted.</td></tr>')

    summary = artifact["issue_clues"].get("summary", {})
    leakage = summary.get("solution_leakage")
    leak_html = (
        f'<p class="muted">Solution leakage: <strong>{_h(leakage)}</strong></p>' if leakage else ""
    )
    return (
        '<section class="panel"><h2>Issue description</h2>'
        f'<div class="issue-text">{issue_html}</div>'
        '<p class="muted" style="margin-top:8px">Highlighted spans are the extracted keywords, colored by clue category. Hover to see the exact category and type.</p>'
        "</section>"
        '<section class="panel"><h2>Extracted keywords</h2>'
        f'{leak_html}'
        '<div style="overflow:auto"><table>'
        '<thead><tr><th>Category</th><th>Type</th><th>Keyword</th><th>Role</th><th>Signal</th><th>Match</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></section>'
    )


def _render_raw(events: list[dict[str, Any]]) -> str:
    cards = []
    for event in events:
        role = str(event.get("role") or "")
        tag_cls = role if role in {"system", "user", "assistant", "tool"} else "assistant"
        kind = event.get("kind") or ""
        tool = event.get("tool_name")
        meta = " · ".join(part for part in [role, str(kind), tool and f"tool: {tool}"] if part)
        cards.append(
            f'<div class="rawcard" id="raw-{_h(event.get("id"))}">'
            '<div class="raw-head">'
            f'<span class="rawtag {tag_cls}">{_h(role or "event")}</span>'
            f'<span>{_h(meta)}</span>'
            f'<span class="fid">{_h(event.get("id"))}</span>'
            "</div>"
            f'<pre class="raw-content">{_h(event.get("content"))}</pre>'
            "</div>"
        )
    if not cards:
        cards.append('<p class="muted">No trajectory events were found.</p>')
    return (
        '<section class="panel"><h2>Raw trajectory</h2>'
        '<p class="muted">Every trajectory event in order. Clicking a snippet in the Graph tab jumps here and highlights the exact text.</p>'
        + "".join(cards)
        + "</section>"
    )


def _render_rules(artifact: dict[str, Any]) -> str:
    rules = artifact.get("rule_based_results", {})

    git_rows = [
        "<tr>"
        f'<td>{_h(cmd.get("event_id"))}</td>'
        f'<td><code>{_h(cmd.get("command"))}</code></td>'
        f'<td>{_h(cmd.get("category"))}</td>'
        "</tr>"
        for cmd in rules.get("git_commands", [])
    ]
    if not git_rows:
        git_rows.append('<tr><td colspan="3" class="muted">No git commands found.</td></tr>')

    grep_rows = []
    for item in rules.get("keyword_grep", []):
        grep_rows.append(
            "<tr>"
            f'<td>{_h(item.get("keyword"))}</td>'
            f'<td>{_h(item.get("category"))}</td>'
            f'<td>{_h(item.get("match_count"))}</td>'
            f'<td>{_h(", ".join(item.get("event_ids", [])) or "—")}</td>'
            "</tr>"
        )
    if not grep_rows:
        grep_rows.append('<tr><td colspan="4" class="muted">No keywords to grep.</td></tr>')

    phase_rows = [
        "<tr>"
        f'<td><span class="pill" style="background:{PHASE_COLORS.get(phase, PHASE_COLORS["General"])}">{_h(phase)}</span></td>'
        f'<td>{_h(count)}</td>'
        "</tr>"
        for phase, count in rules.get("phase_counts", {}).items()
    ]
    if not phase_rows:
        phase_rows.append('<tr><td colspan="2" class="muted">No phases in the graph.</td></tr>')

    return (
        '<section class="panel"><h2>Git commands</h2>'
        '<div style="overflow:auto"><table>'
        '<thead><tr><th>Event</th><th>Command</th><th>Category</th></tr></thead>'
        f'<tbody>{"".join(git_rows)}</tbody></table></div></section>'
        '<section class="panel"><h2>Issue keyword grep</h2>'
        '<p class="muted">Each extracted issue keyword searched across all trajectory events.</p>'
        '<div style="overflow:auto"><table>'
        '<thead><tr><th>Keyword</th><th>Category</th><th>Matches</th><th>Events</th></tr></thead>'
        f'<tbody>{"".join(grep_rows)}</tbody></table></div></section>'
        '<section class="panel"><h2>Phase counts</h2>'
        '<div style="overflow:auto"><table>'
        '<thead><tr><th>Phase</th><th>Node count</th></tr></thead>'
        f'<tbody>{"".join(phase_rows)}</tbody></table></div></section>'
    )


def _render_spt_data(artifact: dict[str, Any]) -> str:
    case = artifact.get("case", {})
    spt = case.get("spt") if isinstance(case.get("spt"), dict) else {}
    entries = spt.get("data") if isinstance(spt.get("data"), list) else []

    def normalize_path(path: Any) -> str:
        if not isinstance(path, str):
            return ""
        normalized = path.strip().replace("\\", "/")
        normalized = normalized.lstrip("./")
        if normalized.startswith("a/") or normalized.startswith("b/"):
            normalized = normalized[2:]
        return normalized

    resource_accesses = (
        artifact.get("deterministic_checks", {})
        .get("resources", {})
        .get("accesses", [])
    )
    spt_file_counts = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        file_path = normalize_path(item.get("file"))
        if not file_path:
            continue
        spt_file_counts[file_path] = spt_file_counts.get(file_path, 0) + 1

    overlap_rows = []
    overlapped_count = 0
    for spt_file, mutation_count in sorted(
        spt_file_counts.items(), key=lambda pair: (-pair[1], pair[0])
    ):
        matches = []
        for access in resource_accesses:
            if not isinstance(access, dict):
                continue
            resource = normalize_path(access.get("resource"))
            if not resource:
                continue
            same = resource == spt_file
            suffix = resource.endswith("/" + spt_file) or spt_file.endswith("/" + resource)
            if same or suffix:
                matches.append(access)

        if matches:
            overlapped_count += 1
            kinds = sorted({str(match.get("kind")) for match in matches if match.get("kind")})
            events = [str(match.get("event_id")) for match in matches if match.get("event_id")]
            first_event = min(events, key=lambda event_id: int(event_id[1:]) if event_id.startswith("E") and event_id[1:].isdigit() else 10**9)
            resources = sorted({normalize_path(match.get("resource")) for match in matches})
            overlap_rows.append(
                "<tr>"
                f"<td>{_h(spt_file)}</td>"
                f"<td>{_h(mutation_count)}</td>"
                f"<td>{_h(', '.join(kinds) or '—')}</td>"
                f"<td>{_h(first_event)}</td>"
                f"<td>{_h(', '.join(resources[:4]))}{_h(' ...' if len(resources) > 4 else '')}</td>"
                "</tr>"
            )
        else:
            overlap_rows.append(
                "<tr>"
                f"<td>{_h(spt_file)}</td>"
                f"<td>{_h(mutation_count)}</td>"
                '<td class="muted">No observed access</td>'
                '<td class="muted">—</td>'
                '<td class="muted">—</td>'
                "</tr>"
            )

    if not overlap_rows:
        overlap_rows.append('<tr><td colspan="5" class="muted">No SPT files available for overlap analysis.</td></tr>')

    meta_rows = [
        ("Metadata available", spt.get("metadata_available")),
        ("Source path", spt.get("source_path")),
        ("Entry count", spt.get("entry_count")),
        ("Applied", spt.get("applied")),
        ("Note", spt.get("note")),
        ("Error", spt.get("error")),
    ]
    meta_table = "".join(
        f"<tr><td>{_h(label)}</td><td>{_h(value if value is not None else '—')}</td></tr>"
        for label, value in meta_rows
    )

    entry_rows = []
    for item in entries[:200]:
        if not isinstance(item, dict):
            continue
        positions = item.get("positions") if isinstance(item.get("positions"), list) else []
        positions_preview = ", ".join(
            f"L{pos.get('line')}:{pos.get('column')}"
            for pos in positions[:8]
            if isinstance(pos, dict)
        )
        if len(positions) > 8:
            positions_preview += f" ... (+{len(positions) - 8} more)"
        entry_rows.append(
            "<tr>"
            f"<td>{_h(item.get('order'))}</td>"
            f"<td>{_h(item.get('transformation'))}</td>"
            f"<td>{_h(item.get('file'))}</td>"
            f"<td>{_h(positions_preview or '—')}</td>"
            "</tr>"
        )
    if not entry_rows:
        entry_rows.append('<tr><td colspan="4" class="muted">No SPT entries found.</td></tr>')

    truncated_note = ""
    if len(entries) > 200:
        truncated_note = (
            f'<p class="muted">Showing first 200 of {len(entries)} SPT entries.</p>'
        )

    raw_spt_json = _embed_json(spt) if spt else "{}"
    return (
        '<section class="panel"><h2>SPT metadata</h2>'
        '<div style="overflow:auto"><table>'
        '<thead><tr><th>Field</th><th>Value</th></tr></thead>'
        f'<tbody>{meta_table}</tbody></table></div></section>'
        '<section class="panel"><h2>SPT-trajectory overlap</h2>'
        f'<p class="muted">{_h(overlapped_count)} of {_h(len(spt_file_counts))} SPT-mutated files were observed in trajectory resource accesses.</p>'
        '<div style="overflow:auto"><table>'
        '<thead><tr><th>SPT file</th><th>Mutation entries</th><th>Observed access kinds</th><th>First event</th><th>Matched trajectory resources</th></tr></thead>'
        f'<tbody>{"".join(overlap_rows)}</tbody></table></div></section>'
        '<section class="panel"><h2>SPT mutations</h2>'
        f"{truncated_note}"
        '<div style="overflow:auto"><table>'
        '<thead><tr><th>Order</th><th>Transformation</th><th>File</th><th>Positions</th></tr></thead>'
        f'<tbody>{"".join(entry_rows)}</tbody></table></div></section>'
        '<section class="panel"><h2>Raw SPT JSON</h2>'
        f'<pre>{_h(raw_spt_json)}</pre>'
        '</section>'
    )


def _render_patch(artifact: dict[str, Any]) -> str:
    case = artifact.get("case", {})
    patch_text = case.get("submission")
    if not isinstance(patch_text, str) or not patch_text.strip():
        patch_text = ""

    patch_summary = (
        artifact.get("deterministic_checks", {})
        .get("patches", {})
        .get("submitted_patch", {})
    )

    summary_rows = []
    if isinstance(patch_summary, dict) and patch_summary:
        summary_rows.append(
            f"<tr><td>Files modified</td><td>{_h(', '.join(patch_summary.get('files_modified', [])) or '—')}</td></tr>"
        )
        summary_rows.append(
            f"<tr><td>Additions</td><td>{_h(patch_summary.get('additions'))}</td></tr>"
        )
        summary_rows.append(
            f"<tr><td>Deletions</td><td>{_h(patch_summary.get('deletions'))}</td></tr>"
        )
        summary_rows.append(
            f"<tr><td>Hunks</td><td>{_h(patch_summary.get('hunks'))}</td></tr>"
        )
    else:
        summary_rows.append('<tr><td colspan="2" class="muted">No patch summary available.</td></tr>')

    patch_body = _h(patch_text) if patch_text else "No submitted patch text was found for this case."
    return (
        '<section class="panel"><h2>Patch summary</h2>'
        '<div style="overflow:auto"><table>'
        '<thead><tr><th>Field</th><th>Value</th></tr></thead>'
        f'<tbody>{"".join(summary_rows)}</tbody></table></div></section>'
        '<section class="panel"><h2>Submitted patch</h2>'
        f'<pre>{patch_body}</pre>'
        '</section>'
    )


MANUAL_EVAL_RUBRIC = [
    (
        "1. Problem Understanding",
        [
            ("pu_spt", "Could the agent identify that SPTs had been applied? If yes, how did this influence subsequent actions?", ["", "Yes", "No", "Unclear", "N/A"]),
            ("pu_clues", "If the issue description contains multiple contextual clues, which ones did the agent prioritize?", None),
            ("pu_git_revert", "Did the agent ever revert to an earlier repository state using Git?", ["", "Yes", "No", "Unclear"]),
        ],
    ),
    (
        "2. Localization",
        [
            ("loc_efficient", "Was the agent able to localize the issue efficiently? Was any increase in file coverage due to injected SPTs or LLM stochasticity?", ["", "Efficient", "Inefficient", "Unclear", "N/A"]),
        ],
    ),
    (
        "3. Planning & Implementation",
        [
            ("plan_first_correct", "Did the agent formulate the correct implementation plan on its first attempt?", ["", "Yes", "No", "Partially", "Unclear"]),
            ("plan_revision", "If not, did it revise or backtrack, and what triggered the revision (failed validation, new evidence, incorrect assumptions)?", None),
        ],
    ),
    (
        "4. Validation",
        [
            ("val_setup", "Did the agent construct the correct validation setup? Were the tests sufficient, and did it iterate on tests after failures?", ["", "Correct", "Insufficient", "None", "Unclear"]),
        ],
    ),
    (
        "5. Final Result",
        [
            ("final_phase", "If the trajectory failed, in which phase did the failure originate?", ["", "Localization", "Planning", "Implementation", "Validation", "Other", "N/A (passed)"]),
            ("final_reason", "Primary reason for the final outcome.", None),
        ],
    ),
    (
        "6. Efficiency & Trajectory Characteristics",
        [
            ("eff_excess_phase", "If the trajectory required substantially more steps than baseline, which phase contributed most to the increase?", ["", "Localization", "Debugging", "Planning", "Patching", "Validation", "Recovery", "General", "N/A"]),
            ("eff_largest_phase", "Which phase occupied the largest portion of the trajectory (time/steps)?", ["", "Localization", "Debugging", "Planning", "Patching", "Validation", "Recovery", "General"]),
        ],
    ),
]


def _render_manual_eval(artifact: dict[str, Any]) -> str:
    case = artifact["case"]
    storage_key = _h(case.get("case_id") or case.get("case_name") or "case")

    sections = []
    for title, items in MANUAL_EVAL_RUBRIC:
        questions = []
        for item_id, question, options in items:
            select_html = ""
            if options is not None:
                option_html = "".join(
                    f'<option value="{_h(opt)}">{_h(opt) or "— select —"}</option>'
                    for opt in options
                )
                select_html = f'<select data-eval name="{_h(item_id)}">{option_html}</select>'
            questions.append(
                '<div class="q">'
                f'<label>{_h(question)}</label>'
                f'{select_html}'
                f'<textarea data-eval name="{_h(item_id)}_notes" placeholder="Notes / justification"></textarea>'
                "</div>"
            )
        sections.append(f'<section class="panel"><h3>{_h(title)}</h3>{"".join(questions)}</section>')

    return (
        f'<form id="eval-form" class="evalform" data-storage-key="{storage_key}" onsubmit="return false">'
        '<section class="panel"><h2>Manual evaluation results</h2>'
        '<p class="muted">Fill this rubric in and click Save. Answers persist in this browser and can be edited later. '
        'Use Download to keep a portable copy, or Load file to restore one.</p>'
        '<div class="evaltoolbar">'
        '<button type="button" class="primary" onclick="saveEval()">Save</button>'
        '<button type="button" onclick="downloadEval()">Download</button>'
        '<label class="jumpbtn" style="display:inline-block;margin:0">'
        '<span class="badge" style="cursor:pointer">Load file</span>'
        '<input type="file" accept="application/json" style="display:none" onchange="loadEvalFile(this)"></label>'
        '<button type="button" onclick="clearEval()">Clear</button>'
        '<span id="eval-status" class="evalstatus">Not yet saved.</span>'
        "</div>"
        '<div class="q"><label>Evaluator</label>'
        '<input type="text" data-eval name="evaluator" placeholder="Your name"></div>'
        "</section>"
        + "".join(sections)
        + '<section class="panel"><h3>Overall notes</h3>'
        '<div class="q"><textarea data-eval name="overall_notes" placeholder="Summary / additional observations"></textarea></div>'
        '<div class="evaltoolbar"><button type="button" class="primary" onclick="saveEval()">Save</button>'
        '<span class="evalstatus">Saved data stays in this browser for this case.</span></div>'
        "</section></form>"
    )


def build_case_html(artifact: dict[str, Any]) -> str:
    case = artifact["case"]
    result = case.get("result") or {}
    resolved = result.get("resolved")
    result_label = "Resolved" if resolved is True else "Unresolved" if resolved is False else "Unknown"
    result_class = "resolved" if resolved is True else "unresolved" if resolved is False else ""
    metadata = case.get("metadata") or {}
    entries = _graph_entries(artifact)
    node_entries = [entry for entry in entries if entry["kind"] == "node"]
    events = artifact.get("events", [])

    graph_data = {
        "nodes": node_entries,
        "context": [entry for entry in entries if entry["kind"] == "context"],
        "events": [
            {
                "id": event.get("id"),
                "content": event.get("content", ""),
            }
            for event in events
        ],
    }

    tabs = [
        ("issue", "Issue description"),
        ("graph", "Graph"),
        ("raw", "Raw trajectory"),
        ("rules", "Rule based results"),
        ("spt", "SPT data"),
        ("patch", "Patch"),
        ("manual", "Manual evaluation"),
    ]
    tab_buttons = "".join(
        f'<button class="{"active" if index == 0 else ""}" data-tab="{name}" '
        f'onclick="switchTab(\'{name}\')">{_h(label)}</button>'
        for index, (name, label) in enumerate(tabs)
    )

    panels = (
        f'<div class="tab-panel active" data-tab="issue">{_render_issue(artifact)}</div>'
        f'<div class="tab-panel" data-tab="graph">{_render_graph(entries)}</div>'
        f'<div class="tab-panel" data-tab="raw">{_render_raw(events)}</div>'
        f'<div class="tab-panel" data-tab="rules">{_render_rules(artifact)}</div>'
        f'<div class="tab-panel" data-tab="spt">{_render_spt_data(artifact)}</div>'
        f'<div class="tab-panel" data-tab="patch">{_render_patch(artifact)}</div>'
        f'<div class="tab-panel" data-tab="manual">{_render_manual_eval(artifact)}</div>'
    )

    body = (
        '<header class="hero">'
        f'<div class="label">AI-assisted trajectory analysis</div><h1>{_h(case.get("case_name"))}</h1>'
        f'<p class="muted">{_h(case.get("case_id"))}</p>'
        '<section class="grid" style="margin-top:12px">'
        f'<div class="card"><div class="label">Result</div><div class="value {result_class}">{result_label}</div></div>'
        f'<div class="card"><div class="label">Format</div><div class="value">{_h(case.get("format"))}</div></div>'
        f'<div class="card"><div class="label">Model</div><div class="value">{_h(metadata.get("model") or "Unknown")}</div></div>'
        f'<div class="card"><div class="label">Graph nodes</div><div class="value">{len(node_entries)}</div></div>'
        "</section></header>"
        f'<div class="tabs">{tab_buttons}</div>'
        + panels
    )
    return _page(
        f"Trajectory analysis: {case.get('case_name')}",
        body,
        _embed_json(graph_data),
    )


def save_case_report(artifact: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)
    (output_dir / "report.html").write_text(build_case_html(artifact), encoding="utf-8")


def save_batch_index(summary: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for case in summary.get("cases", []):
        output_name = case.get("output_directory", "")
        link = (
            f'<a href="{_h(output_name)}/report.html">Open report</a>'
            if case.get("status") == "completed"
            else _h(case.get("error", "Failed"))
        )
        resolved = case.get("resolved")
        resolved_text = "Resolved" if resolved is True else "Unresolved" if resolved is False else "Unknown"
        rows.append(
            "<tr>"
            f'<td>{_h(case.get("case_name"))}</td><td>{_h(case.get("format"))}</td>'
            f'<td>{_h(resolved_text)}</td><td>{_h(case.get("largest_phase", ""))}</td>'
            f'<td>{_h(case.get("event_count", ""))}</td><td>{link}</td></tr>'
        )
    body = (
        '<header class="hero"><div class="label">AI-assisted trajectory analysis</div><h1>Batch report</h1>'
        f'<p>{summary.get("completed", 0)} completed · {summary.get("failed", 0)} failed</p></header>'
        '<section class="panel"><div style="overflow:auto"><table><thead><tr>'
        '<th>Case</th><th>Format</th><th>Result</th><th>Largest phase</th><th>Events</th><th>Report</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>'
    )
    (output_dir / "index.html").write_text(_page("Trajectory analysis batch", body), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["case_name", "case_id", "format", "resolved", "status", "largest_phase", "event_count", "output_directory", "error"]
        )
        for case in summary.get("cases", []):
            writer.writerow(
                [
                    case.get("case_name", ""),
                    case.get("case_id", ""),
                    case.get("format", ""),
                    case.get("resolved", ""),
                    case.get("status", ""),
                    case.get("largest_phase", ""),
                    case.get("event_count", ""),
                    case.get("output_directory", ""),
                    case.get("error", ""),
                ]
            )