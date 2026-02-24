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


def steps_to_graph_data(plan: Plan, estimate_model_fn=None) -> list[dict[str, Any]]:
    """Convert a plan's steps into the data format needed for the timeline graph.

    Args:
        plan: The plan to convert
        estimate_model_fn: Optional function to estimate model for a step (Step -> str)
    """
    steps_data: list[dict[str, Any]] = []
    for s in plan.steps:
        # Filter out self-referencing dependencies
        deps = [d for d in (s.depends_on or []) if d != s.step_num]
        raw_title = s.title or f"Step {s.step_num}"
        title = raw_title[:50] + ("\u2026" if len(raw_title) > 50 else "")

        # Classify task based on keywords
        # Combine title and prompt for comprehensive matching
        text_to_check = ((s.title or "") + " " + (s.prompt or "")).lower()
        
        # Architecture/Design tasks (highest priority - most specific)
        # These are complex, high-level design tasks requiring deep thinking
        arch_kw = (
            # English
            "architect", "architecture", "design system", "system design",
            "reverse engineer", "analyze complex", "redesign", "design pattern",
            "infrastructure", "scalability", "distributed system",
            # Chinese
            "架构", "方案设计", "系统设计", "技术方案", "设计模式",
        )
        
        # Code implementation tasks (medium-high priority)
        # These involve writing, modifying, or testing code
        # Note: Avoid overly broad terms like "编程" (programming) which can match
        # "编程语言" (programming language) in research tasks
        code_kw = (
            # English - implementation
            "implement", "write code", "create function", "build feature",
            "develop code", "program code", "script", "coding",
            # English - modification
            "fix bug", "debug", "refactor", "optimize code",
            "modify code", "update code", "patch",
            # English - testing
            "unit test", "integration test", "write test",
            # English - version control
            "pull request", "github", "git",
            # Chinese - implementation (be specific to avoid false positives)
            "实现", "开发", "写代码", "编写代码", "写程序",
            "写函数", "写脚本", "创建函数",
            # Chinese - modification
            "重构", "修复bug", "调试代码", "优化代码",
            # Chinese - testing
            "单元测试", "集成测试",
        )
        
        # Chinese language tasks (medium priority)
        # These require Chinese language processing or output
        # Use more specific phrases to avoid conflicts
        cn_kw = (
            "中文总结", "中文回答", "中文输出", "用中文",
            "翻译", "摘要", "概括",
            "调研", "搜索", "查找", "收集资料",
        )
        
        # Classification with priority order
        # Special case: if both code and Chinese keywords match, check which is more dominant
        has_arch = any(kw in text_to_check for kw in arch_kw)
        has_code = any(kw in text_to_check for kw in code_kw)
        has_cn = any(kw in text_to_check for kw in cn_kw)
        
        if has_arch:
            category = "架构"
        elif has_code and not has_cn:
            # Pure code task
            category = "代码"
        elif has_cn and not has_code:
            # Pure Chinese task
            category = "中文"
        elif has_code and has_cn:
            # Both matched - count which has more matches
            code_count = sum(1 for kw in code_kw if kw in text_to_check)
            cn_count = sum(1 for kw in cn_kw if kw in text_to_check)
            category = "代码" if code_count > cn_count else "中文"
        else:
            category = "通用"

        entry: dict[str, Any] = {
            "id": s.step_num,
            "title": title,
            "deps": deps,
            "status": s.status.value if hasattr(s.status, "value") else str(s.status),
            "category": category,
        }
        if s.task_id:
            entry["tid"] = s.task_id
        if s.timeout_minutes:
            entry["timeout"] = s.timeout_minutes

        # Use explicit model or estimate it
        model = s.model
        if not model and estimate_model_fn:
            model = estimate_model_fn(s)
        if model:
            entry["model"] = model

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
.node-meta {{ display: flex; align-items: center; gap: 6px; margin-top: 2px; flex-wrap: wrap; }}
.node-tid {{ font-size: 9px; color: #999; font-family: monospace; }}
.node-duration {{ font-size: 9px; color: #2196F3; font-weight: 600; }}
.node-timeout {{ font-size: 9px; color: #FF9800; font-family: monospace; }}
.node-category {{
    font-size: 10px;
    color: #666;
    background: #E8EAF6;
    border: 1px solid #C5CAE9;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
}}
.node-model {{
    font-size: 11px;
    color: #333;
    font-weight: 600;
    font-family: monospace;
    background: #e8f4f8;
    padding: 2px 6px;
    border-radius: 3px;
    border: 1px solid #b3d9e8;
}}
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
const COLS_PER_ROW = 3, svgPad = 60, wrapMargin = 90, topPad = 60;
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
    const timeout = s.timeout
        ? `<span class="node-timeout">\\u23f0${{s.timeout}}m</span>` : '';
    const category = s.category
        ? `<span class="node-category">${{s.category}}</span>` : '';
    const model = s.model
        ? `<span class="node-model">${{s.model.split('/').pop()}}</span>` : '';
    const metaParts = [tid, dur, timeout, category, model].filter(Boolean);
    const meta = metaParts.length
        ? `<div class="node-meta">${{metaParts.join(' ')}}</div>` : '';
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
const rowMaxH = {{}};
const rowY = {{}};
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

    let cumY = phasedStartY;
    for (let r = 0; r < numRows; r++) {{
        rowY[r] = cumY;
        // Count all arrows that cross this gap (including multi-row jumps)
        let gapArrows = 0;
        for (const [key, count] of Object.entries(crossRowArrows)) {{
            const [from, to] = key.split('-').map(Number);
            // If this gap (r to r+1) is between from and to, count it
            if (from <= r && r < to) {{
                gapArrows += count;
            }}
        }}
        const rowGap = baseRowGap + gapArrows * arrowLineSpacing;
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
            // Cross-row: drop down right after source node, travel horizontally
            // in the gap between rows, then rise up to target node
            const gapKey = fromRow + '-' + toRow;
            if (!crossRowArrowIdx[gapKey]) crossRowArrowIdx[gapKey] = 0;
            const idx = crossRowArrowIdx[gapKey];
            crossRowArrowIdx[gapKey]++;
            const vOffset = idx * arrowLineSpacing;
            // Find the gap between rows for horizontal segment
            const fromRowBottom = rowY[fromRow] + rowMaxH[fromRow];
            const toRowTop = rowY[toRow];
            // Place horizontal segments evenly in the gap
            const gapStart = fromRowBottom + (toRowTop - fromRowBottom) * 0.3;
            const gapMid = gapStart + vOffset;
            // Drop down right after source node (x1 + small margin)
            const dropX = x1 + 10 + vOffset;
            // Rise up just before target node
            const riseX = x2 - 15 - vOffset;
            path.setAttribute('d',
                `M${{x1}},${{y1}} H${{dropX}} `
                + `V${{gapMid}} H${{riseX}} V${{y2}} H${{x2}}`);
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
                "() => { const g = document.getElementById('graph');"
                " const c = document.querySelector('.container');"
                " const gw = g ? g.scrollWidth : 0; const gh = g ? g.scrollHeight : 0;"
                " const cw = c.scrollWidth; const ch = c.scrollHeight;"
                " return { w: Math.max(gw, cw), h: Math.max(gh, ch) }; }"
            )
            pad = 32
            need_w = max(1100, dims["w"] + pad * 2)
            need_h = max(600, dims["h"] + pad * 2)
            page.set_viewport_size({"width": need_w, "height": need_h})
            page.wait_for_timeout(200)
            # Precise clip: measure actual content bounds
            box = page.evaluate(
                "() => { const c = document.querySelector('.container');"
                " const r = c.getBoundingClientRect();"
                " return { x: r.x, y: r.y, w: r.width, h: r.height }; }"
            )
            page.screenshot(
                path=output_path,
                clip={
                    "x": max(0, box["x"] - pad),
                    "y": max(0, box["y"] - pad),
                    "width": box["w"] + pad * 2,
                    "height": box["h"] + pad * 2,
                },
            )
            browser.close()
    finally:
        if os.path.exists(html_file):
            os.unlink(html_file)

    return output_path
