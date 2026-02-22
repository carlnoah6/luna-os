"""Plan timeline / dependency graph generation.

Generates an HTML-based timeline visualization for plans and renders it
to PNG using Playwright (optional dependency).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from luna_os.types import Plan


def steps_to_graph_data(plan: Plan) -> list[dict[str, Any]]:
    """Convert a plan's steps into the data format needed for the timeline graph."""
    steps_data: list[dict[str, Any]] = []
    for s in plan.steps:
        # Filter out self-referencing dependencies
        deps = [d for d in (s.depends_on or []) if d != s.step_num]
        raw_title = s.title or f"Step {s.step_num}"
        title = raw_title[:50] + ("\u2026" if len(raw_title) > 50 else "")
        entry: dict[str, Any] = {
            "id": s.step_num,
            "title": title,
            "deps": deps,
            "status": s.status.value if hasattr(s.status, "value") else str(s.status),
        }
        if s.task_id:
            entry["tid"] = s.task_id
        steps_data.append(entry)
    return steps_data


def generate_html(
    steps_data: list[dict[str, Any]],
    title: str = "Plan Timeline",
    subtitle: str = "",
) -> str:
    """Generate timeline HTML from step data with statuses.

    Each step dict: ``{id, title, deps, status, tid?, duration?}``

    - status: pending | running | done | failed
    - tid: optional task ID string
    - duration: optional string for running tasks (e.g. "3m")
    """
    subtitle_html = (
        f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    )
    steps_json = json.dumps(steps_data, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
        Roboto, "Noto Sans CJK SC", "Noto Sans SC", sans-serif;
    background: #fafbfc;
    padding: 32px;
}}
.container {{ position: relative; display: inline-block; }}
.title {{
    font-size: 15px;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 4px;
}}
.subtitle {{
    font-size: 11px;
    color: #888;
    margin-bottom: 20px;
    font-family: monospace;
}}
.graph {{ position: relative; }}
.node {{
    position: absolute;
    display: flex;
    align-items: center;
    gap: 10px;
    background: white;
    border-radius: 10px;
    padding: 10px 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 4px solid #ccc;
    width: 200px;
}}
.node-icon {{
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: white; flex-shrink: 0;
}}
.node-body {{ flex: 1; min-width: 0; overflow: hidden; }}
.node-text {{
    font-size: 12px; color: #333; font-weight: 500; line-height: 1.3;
    word-break: break-word; overflow-wrap: break-word;
    display: -webkit-box; -webkit-line-clamp: 3;
    -webkit-box-orient: vertical; overflow: hidden;
}}
.node-meta {{ display: flex; align-items: center; gap: 6px; margin-top: 2px; }}
.node-tid {{ font-size: 9px; color: #999; font-family: monospace; }}
.node-duration {{ font-size: 9px; color: #2196F3; font-weight: 600; }}
.node-status-icon {{ font-size: 14px; flex-shrink: 0; }}
.node.pending {{ border-left-color: #e0e0e0; }}
.node.pending .node-icon {{ background: #e0e0e0; }}
.node.running {{ border-left-color: #2196F3; background: #f0f7ff; }}
.node.running .node-icon {{ background: #2196F3; }}
.node.done {{ border-left-color: #4CAF50; }}
.node.done .node-icon {{ background: #4CAF50; }}
.node.failed {{ border-left-color: #F44336; }}
.node.failed .node-icon {{ background: #F44336; }}
.node.waiting {{ border-left-color: #FF9800; background: #fff8e1; }}
.node.waiting .node-icon {{ background: #FF9800; }}
.phase-label {{
    position: absolute; font-size: 10px; font-weight: 600;
    color: #666; padding: 2px 10px; border-radius: 10px; background: #eee;
}}
.phase-tag {{ position: absolute; font-size: 9px; color: #999; }}
.section-divider {{
    position: absolute; border-top: 1px dashed #ddd; width: 100%;
}}
svg.arrows {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
</style>
</head>
<body>
<div class="container">
    <div class="title">{title}</div>
    {subtitle_html}
    <div class="graph" id="graph"></div>
</div>
<script>
const steps = {steps_json};
const statusIcons = {{
    pending:'', running:'\\u23f3', done:'\\u2705',
    failed:'\\u274c', waiting:'\\u270b'
}};

// --- Identify independent vs phased steps ---
const dependedOn = new Set();
steps.forEach(s => s.deps.forEach(d => dependedOn.add(d)));
const independent = steps.filter(
    s => s.deps.length === 0 && !dependedOn.has(s.id)
);
const phased = steps.filter(
    s => s.deps.length > 0 || dependedOn.has(s.id)
);

// --- Phase calculation (only for phased steps) ---
function getPhase(id, memo, visiting) {{
    if (memo[id] !== undefined) return memo[id];
    if (visiting.has(id)) {{ memo[id] = 0; return 0; }}
    visiting.add(id);
    const s = phased.find(s => s.id === id);
    if (!s || !s.deps.length) {{ memo[id] = 0; return 0; }}
    const validDeps = s.deps.filter(d => d !== id);
    if (!validDeps.length) {{ memo[id] = 0; return 0; }}
    memo[id] = Math.max(
        ...validDeps.map(d => getPhase(d, memo, visiting))
    ) + 1;
    return memo[id];
}}
const phaseMemo = {{}};
phased.forEach(s => getPhase(s.id, phaseMemo, new Set()));

const maxPhase = phased.length > 0
    ? Math.max(...phased.map(s => phaseMemo[s.id] || 0)) : -1;
const phaseGroups = {{}};
phased.forEach(s => {{
    const p = phaseMemo[s.id] || 0;
    if (!phaseGroups[p]) phaseGroups[p] = [];
    phaseGroups[p].push(s);
}});

const nodeW = 200, gapY = 10, phaseGap = 90, labelH = 36;
const COLS_PER_ROW = 3, svgPad = 60, wrapMargin = 30, topPad = 60;
const arrowLineSpacing = 8, baseRowGap = 50;
const graph = document.getElementById('graph');

// --- Pass 1: create nodes off-screen to measure heights ---
const nodeElements = {{}};
const measuredH = {{}};

steps.forEach(s => {{
    const el = document.createElement('div');
    el.className = `node ${{s.status}}`;
    el.style.cssText = `position:absolute;left:-9999px;top:0;width:${{nodeW}}px;`;
    const icon = statusIcons[s.status] || '';
    const tid = s.tid
        ? `<span class="node-tid">${{s.tid}}</span>` : '';
    const dur = (s.status === 'running' && s.duration)
        ? `<span class="node-duration">\\u23f1${{s.duration}}</span>` : '';
    const meta = (tid || dur)
        ? `<div class="node-meta">${{tid}}${{dur}}</div>` : '';
    el.innerHTML = `
        <div class="node-icon">S${{s.id}}</div>
        <div class="node-body">
            <div class="node-text">${{s.title}}</div>
            ${{meta}}
        </div>
        ${{icon ? `<div class="node-status-icon">${{icon}}</div>` : ''}}
    `;
    graph.appendChild(el);
    nodeElements[s.id] = el;
}});
steps.forEach(s => {{ measuredH[s.id] = nodeElements[s.id].offsetHeight; }});

// --- Layout independent steps at the top ---
const nodePositions = {{}};
let totalW = 0, totalH = 0;
let phasedStartY = topPad;

if (independent.length > 0) {{
    const secLabel = document.createElement('div');
    secLabel.className = 'phase-label';
    secLabel.style.cssText = `left:${{wrapMargin}}px;top:${{topPad}}px;`;
    secLabel.textContent = 'Independent';
    graph.appendChild(secLabel);

    const secTag = document.createElement('div');
    secTag.className = 'phase-tag';
    secTag.style.cssText = `left:${{wrapMargin + 80}}px;top:${{topPad + 2}}px;`;
    secTag.textContent = 'no dependencies';
    graph.appendChild(secTag);

    let curX = wrapMargin, curY = topPad + labelH;
    let rowH = 0, col = 0;
    independent.forEach(s => {{
        if (col >= COLS_PER_ROW) {{
            curX = wrapMargin;
            curY += rowH + gapY;
            rowH = 0;
            col = 0;
        }}
        const h = measuredH[s.id];
        const el = nodeElements[s.id];
        el.style.cssText = `position:absolute;left:${{curX}}px;`
            + `top:${{curY}}px;width:${{nodeW}}px;`;
        nodePositions[s.id] = {{
            cx: curX + nodeW, cy: curY + h / 2,
            lx: curX, ly: curY + h / 2,
            top: curY, bottom: curY + h,
            midX: curX + nodeW / 2, phase: -1,
        }};
        rowH = Math.max(rowH, h);
        totalW = Math.max(totalW, curX + nodeW);
        curX += nodeW + 16;
        col++;
    }});
    phasedStartY = curY + rowH + baseRowGap + 10;
    totalH = Math.max(totalH, curY + rowH + gapY);

    // Divider line
    if (phased.length > 0) {{
        const div = document.createElement('div');
        div.className = 'section-divider';
        div.style.cssText = `top:${{phasedStartY - 15}}px;`
            + `width:${{totalW + wrapMargin}}px;left:${{wrapMargin}}px;`;
        graph.appendChild(div);
    }}
}}

// --- Layout phased steps ---
if (phased.length > 0) {{
    function getColHeight(group) {{
        let h = labelH;
        group.forEach((s, i) => {{
            h += measuredH[s.id] + (i > 0 ? gapY : 0);
        }});
        return h;
    }}

    const numRows = Math.floor(maxPhase / COLS_PER_ROW) + 1;
    const rowMaxH = {{}};
    for (let r = 0; r < numRows; r++) {{
        let maxH = 0;
        for (let c = 0; c < COLS_PER_ROW; c++) {{
            const p = r * COLS_PER_ROW + c;
            if (p > maxPhase) break;
            maxH = Math.max(maxH, getColHeight(phaseGroups[p] || []));
        }}
        rowMaxH[r] = maxH;
    }}

    const crossRowArrows = {{}};
    phased.forEach(s => {{
        const toRow = Math.floor((phaseMemo[s.id] || 0) / COLS_PER_ROW);
        s.deps.forEach(depId => {{
            const fromRow = Math.floor(
                (phaseMemo[depId] || 0) / COLS_PER_ROW
            );
            if (fromRow !== toRow) {{
                const gapKey = Math.min(fromRow, toRow)
                    + '-' + Math.max(fromRow, toRow);
                crossRowArrows[gapKey] =
                    (crossRowArrows[gapKey] || 0) + 1;
            }}
        }});
    }});

    const rowY = {{}};
    let cumY = phasedStartY;
    for (let r = 0; r < numRows; r++) {{
        rowY[r] = cumY;
        const directKey = r + '-' + (r + 1);
        const directArrows = crossRowArrows[directKey] || 0;
        const rowGap = baseRowGap + directArrows * arrowLineSpacing;
        cumY += rowMaxH[r] + rowGap;
    }}
    totalH = Math.max(totalH, cumY);

    for (let p = 0; p <= maxPhase; p++) {{
        const group = phaseGroups[p] || [];
        const row = Math.floor(p / COLS_PER_ROW);
        const colInRow = p % COLS_PER_ROW;
        const colX = wrapMargin + colInRow * (nodeW + phaseGap);
        const baseY = rowY[row];
        const thisColH = getColHeight(group);
        const maxH = rowMaxH[row];
        const offsetY = (maxH - thisColH) / 2;

        const lbl = document.createElement('div');
        lbl.className = 'phase-label';
        lbl.style.cssText = `left:${{colX}}px;top:${{baseY + offsetY}}px;`;
        lbl.textContent = `Phase ${{p + 1}}`;
        graph.appendChild(lbl);

        const tag = document.createElement('div');
        tag.className = 'phase-tag';
        tag.style.cssText =
            `left:${{colX + 62}}px;top:${{baseY + offsetY + 2}}px;`;
        tag.textContent = group.length > 1
            ? 'parallel' : (p > 0 ? 'sequential' : '');
        graph.appendChild(tag);

        let curY = baseY + offsetY + labelH;
        group.forEach(s => {{
            const h = measuredH[s.id];
            const el = nodeElements[s.id];
            el.style.cssText = `position:absolute;left:${{colX}}px;`
                + `top:${{curY}}px;width:${{nodeW}}px;`;
            nodePositions[s.id] = {{
                cx: colX + nodeW, cy: curY + h / 2,
                lx: colX, ly: curY + h / 2,
                top: curY, bottom: curY + h,
                midX: colX + nodeW / 2,
                phase: phaseMemo[s.id],
            }};
            curY += h + gapY;
        }});
        totalW = Math.max(totalW, colX + nodeW);
    }}
}}

graph.style.width = (totalW + wrapMargin + svgPad) + 'px';
graph.style.height = (totalH + svgPad + topPad) + 'px';

// --- Draw arrows ---
const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
svg.setAttribute('class', 'arrows');
svg.setAttribute('width', totalW + wrapMargin + svgPad);
svg.setAttribute('height', totalH + svgPad + topPad);

const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
function makeMarker(id, fill) {{
    const m = document.createElementNS(
        'http://www.w3.org/2000/svg', 'marker'
    );
    m.setAttribute('id', id);
    m.setAttribute('markerWidth', '8');
    m.setAttribute('markerHeight', '6');
    m.setAttribute('refX', '8');
    m.setAttribute('refY', '3');
    m.setAttribute('orient', 'auto');
    const poly = document.createElementNS(
        'http://www.w3.org/2000/svg', 'polygon'
    );
    poly.setAttribute('points', '0 0, 8 3, 0 6');
    poly.setAttribute('fill', fill);
    m.appendChild(poly);
    return m;
}}
defs.appendChild(makeMarker('arrowhead', '#bbb'));
defs.appendChild(makeMarker('arrowhead-done', '#4CAF50'));
svg.appendChild(defs);

const stepMap = {{}};
steps.forEach(s => stepMap[s.id] = s);
const crossRowArrowIdx = {{}};
const allNodeBoxes = [];
steps.forEach(s => {{
    const pos = nodePositions[s.id];
    if (pos) allNodeBoxes.push({{
        id: s.id, phase: pos.phase,
        left: pos.lx, right: pos.cx,
        top: pos.top, bottom: pos.bottom,
    }});
}});

let bypassAboveIdx = 0, bypassBelowIdx = 0;

steps.forEach(s => {{
    s.deps.forEach(depId => {{
        const from = nodePositions[depId];
        const to = nodePositions[s.id];
        if (!from || !to) return;
        const depStep = stepMap[depId];
        const isDone = depStep && depStep.status === 'done';
        const strokeColor = isDone ? '#4CAF50' : '#ddd';
        const strokeWidth = isDone ? '2' : '1.5';
        const markerEnd = isDone
            ? 'url(#arrowhead-done)' : 'url(#arrowhead)';
        const path = document.createElementNS(
            'http://www.w3.org/2000/svg', 'path'
        );
        const fromPhase = from.phase;
        const toPhase = to.phase;
        const fromRow = fromPhase >= 0
            ? Math.floor(fromPhase / COLS_PER_ROW) : -1;
        const toRow = toPhase >= 0
            ? Math.floor(toPhase / COLS_PER_ROW) : -1;

        if (fromRow === toRow && fromRow >= 0) {{
            const intermediateNodes = allNodeBoxes.filter(n =>
                n.phase > fromPhase && n.phase < toPhase
                && Math.floor(n.phase / COLS_PER_ROW) === fromRow
            );
            const x1 = from.cx + 2, y1 = from.cy;
            const x2 = to.lx - 2, y2 = to.ly;
            if (intermediateNodes.length === 0) {{
                const midX = (x1 + x2) / 2;
                path.setAttribute('d',
                    `M${{x1}},${{y1}} C${{midX}},${{y1}} `
                    + `${{midX}},${{y2}} ${{x2}},${{y2}}`);
            }} else {{
                const bypassSpacing = 18;
                const rCenter = (from.top + to.bottom) / 2;
                const avgY = (from.cy + to.ly) / 2;
                if (avgY <= rCenter) {{
                    const minTop = Math.min(
                        ...intermediateNodes.map(n => n.top)
                    );
                    const arcY = minTop - 35
                        - bypassAboveIdx * bypassSpacing;
                    bypassAboveIdx++;
                    path.setAttribute('d',
                        `M${{x1}},${{y1}} C${{x1}},${{arcY}} `
                        + `${{x2}},${{arcY}} ${{x2}},${{y2}}`);
                }} else {{
                    const maxBot = Math.max(
                        ...intermediateNodes.map(n => n.bottom)
                    );
                    const arcY = maxBot + 35
                        + bypassBelowIdx * bypassSpacing;
                    bypassBelowIdx++;
                    path.setAttribute('d',
                        `M${{x1}},${{y1}} C${{x1}},${{arcY}} `
                        + `${{x2}},${{arcY}} ${{x2}},${{y2}}`);
                }}
            }}
        }} else {{
            const x1 = from.cx + 2, y1 = from.cy;
            const x2 = to.lx - 2, y2 = to.ly;
            // Cross-row: compact polyline that stays close to the nodes
            const gapKey = fromRow + '-' + toRow;
            if (!crossRowArrowIdx[gapKey]) crossRowArrowIdx[gapKey] = 0;
            const vOffset = crossRowArrowIdx[gapKey] * arrowLineSpacing;
            crossRowArrowIdx[gapKey]++;
            // Find the gap between rows — route arrows above the second row
            const fromBottom = from.bottom || from.cy + 20;
            const toTop = to.top || to.cy - 20;
            // Place the horizontal segment just below the first row
            // (30% into the gap, not 50%) to avoid overlapping Phase labels
            const gapMid = fromBottom + (toTop - fromBottom) * 0.25;
            // Stay close: extend only 20px beyond the rightmost node
            const margin = 20 + vOffset;
            const turnX = Math.max(x1, x2) + margin;
            // Offset the descent line further left to leave room for arrowheads
            const descentX = x2 - 20 - vOffset;
            path.setAttribute('d',
                `M${{x1}},${{y1}} H${{turnX}} `
                + `V${{gapMid}} H${{descentX}} V${{y2}} H${{x2}}`);
        }}
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', strokeColor);
        path.setAttribute('stroke-width', strokeWidth);
        path.setAttribute('marker-end', markerEnd);
        svg.appendChild(path);
    }});
}});

graph.insertBefore(svg, graph.firstChild);
</script>
</body>
</html>"""


def render_png(html_content: str, output_path: str) -> str:
    """Render the timeline HTML to a PNG file using Playwright.

    Requires ``playwright`` to be installed with Chromium.
    """
    from playwright.sync_api import sync_playwright

    html_file = tempfile.mktemp(suffix=".html")
    with open(html_file, "w") as f:
        f.write(html_content)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])
            page = browser.new_page(
                viewport={"width": 1100, "height": 600},
                device_scale_factor=2,
            )
            page.goto(f"file://{html_file}")
            page.wait_for_timeout(300)
            dims = page.evaluate(
                "() => { const c = document.querySelector('.container');"
                " return { w: c.scrollWidth, h: c.scrollHeight }; }"
            )
            need_w = max(1100, dims["w"] + 80)
            need_h = max(600, dims["h"] + 80)
            page.set_viewport_size({"width": need_w, "height": need_h})
            page.wait_for_timeout(200)
            container = page.query_selector(".container")
            box = container.bounding_box()
            page.screenshot(
                path=output_path,
                clip={
                    "x": box["x"] - 16,
                    "y": box["y"] - 16,
                    "width": box["width"] + 32,
                    "height": box["height"] + 32,
                },
            )
            browser.close()
    finally:
        if os.path.exists(html_file):
            os.unlink(html_file)

    return output_path
