#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Generate an HTML visualization of the TilingLinalg IR lowering pipeline.

Usage:
    python gen_irflow_html.py                    # default: reads irflow.json, writes irflow.html
    python gen_irflow_html.py -i flow.json       # custom input
    python gen_irflow_html.py -o output.html     # custom output
"""

import argparse
import json
import os
import sys


def load_flow(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def dialect_color(dialects: dict, dialect_id: str) -> str:
    return dialects.get(dialect_id, {}).get("color", "#888")


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_html(data: dict) -> str:
    dialects = data.get("dialects", {})

    # Build dialect legend HTML
    legend_items = []
    for did, info in dialects.items():
        c = info["color"]
        legend_items.append(
            f'<span class="legend-item">'
            f'<span class="legend-dot" style="background:{c}"></span>'
            f'{escape_html(info["name"])}'
            f'</span>'
        )
    legend_html = "\n".join(legend_items)

    # Build pipeline sections
    pipeline_sections = []
    for pipeline in data["pipelines"]:
        pid = pipeline["id"]
        pname = escape_html(pipeline["name"])
        pdesc = escape_html(pipeline["description"])
        outputs = ", ".join(f"<code>{o}</code>" for o in pipeline.get("outputs", []))

        phases_html = []
        for phase in pipeline["phases"]:
            phase_id = phase["id"]
            phase_name = escape_html(phase["name"])
            phase_desc = escape_html(phase.get("description", ""))
            branch_point = phase.get("branch_point", "")

            # Build stage nodes
            stage_nodes = []
            stages = phase.get("stages", [])
            for i, st in enumerate(stages):
                color_in = dialect_color(dialects, st["dialect_in"])
                color_out = dialect_color(dialects, st["dialect_out"])
                label = escape_html(st["label"])
                pass_name = escape_html(st["pass"])
                desc = escape_html(st.get("description", ""))
                ir_file = escape_html(st.get("ir_file", ""))
                stage_num = st["stage"]

                # Use gradient if dialect changes
                if color_in != color_out:
                    bg = f"linear-gradient(135deg, {color_in} 0%, {color_out} 100%)"
                else:
                    bg = color_in

                node_html = f'''<div class="stage-node" style="background: {bg}" title="{desc}">
                    <div class="stage-num">Stage {stage_num}</div>
                    <div class="stage-label">{label}</div>
                    <div class="stage-pass">{pass_name}</div>
                    <div class="stage-dialect">{escape_html(st["dialect_in"])} &rarr; {escape_html(st["dialect_out"])}</div>
                    <div class="stage-ir">{ir_file}</div>
                </div>'''
                stage_nodes.append(node_html)

                # Add arrow between stages (not after the last one)
                if i < len(stages) - 1:
                    stage_nodes.append('<div class="arrow">&darr;</div>')

            # Add emit node
            emit = phase.get("emit")
            if emit:
                stage_nodes.append('<div class="arrow">&darr;</div>')
                out_file = escape_html(emit["output_file"])
                emit_desc = escape_html(emit.get("description", ""))
                stage_nodes.append(
                    f'<div class="emit-node" title="{emit_desc}">'
                    f'<div class="emit-method">{escape_html(emit["method"])}</div>'
                    f'<div class="emit-output">{out_file}</div>'
                    f'</div>'
                )

            stages_html = "\n".join(stage_nodes)

            branch_html = ""
            if branch_point:
                branch_html = f'<div class="branch-label">{escape_html(branch_point)}</div>'

            phases_html.append(f'''
                <div class="phase" id="phase-{pid}-{phase_id}">
                    <div class="phase-header">
                        <h3>{phase_name}</h3>
                        <p>{phase_desc}</p>
                        {branch_html}
                    </div>
                    <div class="stages-column">
                        {stages_html}
                    </div>
                </div>
            ''')

        # Determine layout: if pipeline has multiple phases (host+kernel branches), use side-by-side
        has_branches = len(pipeline["phases"]) > 1
        if has_branches:
            # Separate shared from branch phases
            shared_html = ""
            branch_phases = []
            for ph_html, phase in zip(phases_html, pipeline["phases"]):
                if phase["id"] == "shared":
                    shared_html = ph_html
                else:
                    branch_phases.append(ph_html)

            branch_container = '<div class="branch-container">' + "\n".join(branch_phases) + "</div>"
            # Add a fork arrow between shared and branches
            fork_arrow = '<div class="fork-arrow"><div class="fork-label">Module Clone (Branch Point)</div><div class="fork-lines"><span>&swarr;</span><span>&searr;</span></div></div>'
            inner_html = shared_html + fork_arrow + branch_container
        else:
            inner_html = "\n".join(phases_html)

        pipeline_sections.append(f'''
            <div class="pipeline" id="pipeline-{pid}">
                <div class="pipeline-header">
                    <h2>{pname}</h2>
                    <p>{pdesc}</p>
                    <div class="pipeline-outputs">Outputs: {outputs}</div>
                </div>
                <div class="pipeline-body">
                    {inner_html}
                </div>
            </div>
        ''')

    pipelines_html = "\n".join(pipeline_sections)

    # Test commands
    test_cmds = data.get("test_commands", {})
    test_html = ""
    if test_cmds:
        rows = []
        for cmd_name, cmd_val in test_cmds.items():
            rows.append(f"<tr><td><code>{escape_html(cmd_name)}</code></td><td><code>{escape_html(cmd_val)}</code></td></tr>")
        test_html = f'''
        <div class="test-commands">
            <h2>Test Commands</h2>
            <table>
                <tr><th>Mode</th><th>Command</th></tr>
                {"".join(rows)}
            </table>
        </div>
        '''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape_html(data.get("title", "IR Flow"))}</title>
<style>
:root {{
    --bg: #1a1a2e;
    --bg2: #16213e;
    --fg: #e0e0e0;
    --fg2: #a0a0a0;
    --border: #2a2a4a;
    --accent: #4A90D9;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--fg);
    padding: 2rem;
    min-height: 100vh;
}}
h1 {{
    text-align: center;
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    color: #fff;
}}
.subtitle {{
    text-align: center;
    color: var(--fg2);
    margin-bottom: 2rem;
    font-size: 0.95rem;
}}
.legend {{
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 2rem;
    padding: 1rem;
    background: var(--bg2);
    border-radius: 8px;
    border: 1px solid var(--border);
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
}}
.legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
}}

/* Pipeline */
.pipeline {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 2rem;
    overflow: hidden;
}}
.pipeline-header {{
    padding: 1.5rem;
    border-bottom: 1px solid var(--border);
}}
.pipeline-header h2 {{
    font-size: 1.3rem;
    color: #fff;
    margin-bottom: 0.3rem;
}}
.pipeline-header p {{
    color: var(--fg2);
    font-size: 0.9rem;
}}
.pipeline-outputs {{
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: var(--accent);
}}
.pipeline-outputs code {{
    background: rgba(74, 144, 217, 0.15);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85rem;
}}
.pipeline-body {{
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

/* Phase */
.phase {{
    margin-bottom: 1rem;
    min-width: 280px;
}}
.phase-header {{
    text-align: center;
    margin-bottom: 1rem;
}}
.phase-header h3 {{
    font-size: 1.05rem;
    color: #ddd;
    margin-bottom: 0.2rem;
}}
.phase-header p {{
    font-size: 0.8rem;
    color: var(--fg2);
}}
.branch-label {{
    display: inline-block;
    margin-top: 0.3rem;
    padding: 2px 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    font-size: 0.75rem;
    color: var(--fg2);
}}

/* Stages column */
.stages-column {{
    display: flex;
    flex-direction: column;
    align-items: center;
}}

/* Stage node */
.stage-node {{
    width: 300px;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    color: #fff;
    text-align: center;
    cursor: default;
    transition: transform 0.15s, box-shadow 0.15s;
    position: relative;
}}
.stage-node:hover {{
    transform: scale(1.03);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    z-index: 2;
}}
.stage-num {{
    font-size: 0.65rem;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.stage-label {{
    font-size: 1rem;
    font-weight: 600;
    margin: 0.15rem 0;
}}
.stage-pass {{
    font-size: 0.7rem;
    opacity: 0.8;
    font-family: 'Fira Code', 'Consolas', monospace;
}}
.stage-dialect {{
    font-size: 0.7rem;
    margin-top: 0.2rem;
    opacity: 0.85;
}}
.stage-ir {{
    font-size: 0.6rem;
    opacity: 0.55;
    font-family: monospace;
    margin-top: 0.15rem;
    word-break: break-all;
}}

/* Arrows */
.arrow {{
    font-size: 1.4rem;
    color: var(--fg2);
    margin: 0.2rem 0;
    user-select: none;
}}

/* Emit node */
.emit-node {{
    width: 260px;
    padding: 0.7rem 1rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #2c3e50, #34495e);
    border: 2px dashed #5dade2;
    text-align: center;
}}
.emit-method {{
    font-size: 0.7rem;
    color: #5dade2;
    font-family: monospace;
    margin-bottom: 0.2rem;
}}
.emit-output {{
    font-size: 1.1rem;
    font-weight: 700;
    color: #5dade2;
}}

/* Fork arrow for branch point */
.fork-arrow {{
    text-align: center;
    margin: 0.8rem 0;
    color: var(--fg2);
}}
.fork-label {{
    font-size: 0.8rem;
    color: #f39c12;
    margin-bottom: 0.3rem;
    font-weight: 600;
}}
.fork-lines {{
    font-size: 2rem;
    display: flex;
    justify-content: center;
    gap: 4rem;
}}

/* Branch container for side-by-side phases */
.branch-container {{
    display: flex;
    gap: 2rem;
    justify-content: center;
    flex-wrap: wrap;
}}

/* Test commands */
.test-commands {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
}}
.test-commands h2 {{
    font-size: 1.1rem;
    margin-bottom: 0.8rem;
    color: #fff;
}}
.test-commands table {{
    width: 100%;
    border-collapse: collapse;
}}
.test-commands th, .test-commands td {{
    padding: 0.5rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}}
.test-commands th {{
    color: var(--fg2);
    font-weight: 600;
}}
.test-commands code {{
    background: rgba(255,255,255,0.06);
    padding: 2px 6px;
    border-radius: 4px;
}}

/* Tab navigation */
.tab-nav {{
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}}
.tab-btn {{
    padding: 0.5rem 1.2rem;
    border: 1px solid var(--border);
    background: var(--bg2);
    color: var(--fg2);
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.85rem;
    transition: all 0.2s;
}}
.tab-btn:hover {{
    background: rgba(74, 144, 217, 0.15);
    color: var(--fg);
}}
.tab-btn.active {{
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
}}
.pipeline.hidden {{
    display: none;
}}

@media (max-width: 700px) {{
    body {{ padding: 1rem; }}
    .stage-node {{ width: 240px; }}
    .emit-node {{ width: 220px; }}
    .branch-container {{ flex-direction: column; align-items: center; }}
}}
</style>
</head>
<body>

<h1>{escape_html(data.get("title", "IR Flow"))}</h1>
<p class="subtitle">{escape_html(data.get("description", ""))}</p>

<div class="legend">
    {legend_html}
</div>

<div class="tab-nav" id="tabNav"></div>

{pipelines_html}

{test_html}

<script>
// Tab navigation
(function() {{
    const pipelines = document.querySelectorAll('.pipeline');
    const nav = document.getElementById('tabNav');
    if (pipelines.length <= 1) return;

    const names = [];
    pipelines.forEach((p, i) => {{
        const h2 = p.querySelector('.pipeline-header h2');
        const name = h2 ? h2.textContent : 'Pipeline ' + (i+1);
        names.push(name);
    }});

    // Add "All" button
    const allBtn = document.createElement('button');
    allBtn.className = 'tab-btn active';
    allBtn.textContent = 'All';
    allBtn.onclick = () => {{
        pipelines.forEach(p => p.classList.remove('hidden'));
        nav.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        allBtn.classList.add('active');
    }};
    nav.appendChild(allBtn);

    names.forEach((name, i) => {{
        const btn = document.createElement('button');
        btn.className = 'tab-btn';
        btn.textContent = name.replace(/ Pipeline.*/, '');
        btn.onclick = () => {{
            pipelines.forEach((p, j) => {{
                p.classList.toggle('hidden', j !== i);
            }});
            nav.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }};
        nav.appendChild(btn);
    }});
}})();
</script>

</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML visualization of IR lowering flow")
    parser.add_argument("-i", "--input", default=None, help="Input JSON file (default: irflow.json in same directory)")
    parser.add_argument("-o", "--output", default=None, help="Output HTML file (default: irflow.html in same directory)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input or os.path.join(script_dir, "irflow.json")
    output_path = args.output or os.path.join(script_dir, "irflow.html")

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    data = load_flow(input_path)
    html = build_html(data)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
