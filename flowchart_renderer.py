"""
DocForge AI — flowchart_renderer.py  v1.0
Converts a Mermaid flowchart TD block into a high-quality PNG image
using only matplotlib + Python stdlib. No Node.js, no external APIs.
"""

import re
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Tuple

C_RECT      = "#4A7CC7"
C_RECT_TXT  = "#FFFFFF"
C_DIA       = "#5B9BD5"
C_DIA_TXT   = "#FFFFFF"
C_ROUND     = "#7B3F8C"
C_ROUND_TXT = "#FFFFFF"
C_ARROW     = "#555555"
C_SHADOW    = "#DDDDDD"
C_BG        = "#FFFFFF"


def parse_mermaid(text: str) -> Tuple[Dict, List]:
    nodes: Dict[str, dict] = {}
    edges: List[Tuple]     = []

    rounded_re = re.compile(r'(\w+)\(\[([^\]\)]+)\]\)')
    diamond_re = re.compile(r'(\w+)\{([^\}]+)\}')
    rect_re    = re.compile(r'(\w+)\[([^\]]+)\]')
    label_edge = re.compile(r'(\w+)\s*-+>\|([^|]+)\|\s*(\w+)')
    plain_edge = re.compile(r'(\w+)\s*-+>\s*(\w+)')

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('flowchart') or line.startswith('graph'):
            continue
        for m in rounded_re.finditer(line):
            nid = m.group(1)
            if nid not in nodes:
                nodes[nid] = {'label': m.group(2).strip(), 'shape': 'rounded'}
        for m in diamond_re.finditer(line):
            nid = m.group(1)
            if nid not in nodes:
                nodes[nid] = {'label': m.group(2).strip(), 'shape': 'diamond'}
        for m in rect_re.finditer(line):
            nid = m.group(1)
            if nid not in nodes:
                nodes[nid] = {'label': m.group(2).strip(), 'shape': 'rect'}

        labeled_pairs = set()
        for m in label_edge.finditer(line):
            edges.append((m.group(1), m.group(3), m.group(2).strip()))
            labeled_pairs.add((m.group(1), m.group(3)))
        for m in plain_edge.finditer(line):
            pair = (m.group(1), m.group(2))
            if pair not in labeled_pairs:
                edges.append((m.group(1), m.group(2), ''))
                labeled_pairs.add(pair)

    return nodes, edges


def _topological_layout(nodes: Dict, edges: List) -> Dict[str, Tuple[float, float]]:
    children = {n: [] for n in nodes}
    parents  = {n: [] for n in nodes}
    for (src, dst, _) in edges:
        if src in children and dst in children:
            children[src].append(dst)
            parents[dst].append(src)

    layer = {}
    queue = [n for n in nodes if not parents[n]]
    if not queue:
        queue = list(nodes.keys())[:1]
    for n in queue:
        layer[n] = 0

    changed, iterations = True, 0
    while changed and iterations < 100:
        changed = False
        iterations += 1
        for (src, dst, _) in edges:
            if src in layer:
                nl = layer[src] + 1
                if layer.get(dst, -1) < nl:
                    layer[dst] = nl
                    changed = True

    for n in nodes:
        if n not in layer:
            layer[n] = 0

    layers = {}
    for n, l in layer.items():
        layers.setdefault(l, []).append(n)

    pos = {}
    Y_STEP, X_STEP = 2.4, 2.8

    for l_idx in sorted(layers.keys()):
        layer_nodes = layers[l_idx]
        n = len(layer_nodes)
        x_start = -(n - 1) * X_STEP / 2
        y = -l_idx * Y_STEP
        for i, node_id in enumerate(layer_nodes):
            pos[node_id] = (x_start + i * X_STEP, y)

    return pos


def _wrap(text: str, max_chars: int = 16) -> str:
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current = (current + " " + w).strip()
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return "\n".join(lines)


def _draw_rect(ax, x, y, w, h, label, color, txt_color):
    ax.add_patch(FancyBboxPatch((x-w/2+0.04, y-h/2-0.06), w, h,
        boxstyle="round,pad=0.1", linewidth=0, facecolor=C_SHADOW, zorder=1))
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.1", linewidth=1.5,
        edgecolor="#2E4057", facecolor=color, zorder=2))
    ax.text(x, y, _wrap(label), ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=txt_color,
            zorder=3, linespacing=1.35)


def _draw_diamond(ax, x, y, w, h, label, color, txt_color):
    hw, hh = w/2, h/2
    ax.add_patch(plt.Polygon(
        [[x+0.04, y+hh-0.06],[x+hw+0.04,y-0.06],[x+0.04,y-hh-0.06],[x-hw+0.04,y-0.06]],
        closed=True, facecolor=C_SHADOW, linewidth=0, zorder=1))
    ax.add_patch(plt.Polygon(
        [[x, y+hh],[x+hw,y],[x,y-hh],[x-hw,y]],
        closed=True, facecolor=color, edgecolor="#2E4057", linewidth=1.5, zorder=2))
    ax.text(x, y, _wrap(label, 12), ha='center', va='center',
            fontsize=8, fontweight='bold', color=txt_color,
            zorder=3, linespacing=1.3)


def _draw_pill(ax, x, y, w, h, label, color, txt_color):
    ax.add_patch(FancyBboxPatch((x-w/2+0.04, y-h/2-0.06), w, h,
        boxstyle="round,pad=0.28", linewidth=0, facecolor=C_SHADOW, zorder=1))
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.28", linewidth=1.5,
        edgecolor="#4A1460", facecolor=color, zorder=2))
    ax.text(x, y, label, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=txt_color, zorder=3)


def _draw_arrow(ax, x1, y1, x2, y2, label='', bend=False):
    rad = 0.4 if bend else 0.0
    ax.annotate('', xy=(x2, y2+0.38), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.6,
            connectionstyle=f"arc3,rad={rad}"), zorder=4)
    if label:
        mx = (x1+x2)/2 + (0.3 if bend else 0.12)
        my = (y1+y2)/2
        ax.text(mx, my, label, fontsize=7.5, color='#333333',
            ha='center', va='center', zorder=5,
            bbox=dict(facecolor='white', edgecolor='#BBBBBB',
                      boxstyle='round,pad=0.2', linewidth=0.8))


def mermaid_to_png_bytes(mermaid_text: str, title: str = "", dpi: int = 180) -> bytes:
    """
    Convert a Mermaid flowchart TD block to PNG bytes.
    Args:
        mermaid_text: Raw mermaid text (with or without ```mermaid fences)
        title:        Optional title above the chart
        dpi:          Output resolution (180 recommended for Word)
    Returns:
        bytes: PNG image bytes ready for python-docx doc.add_picture()
    """
    clean = re.sub(r'```mermaid\s*', '', mermaid_text)
    clean = re.sub(r'```\s*$', '', clean, flags=re.MULTILINE)

    nodes, edges = parse_mermaid(clean)

    if not nodes:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "[ Process Flow Diagram ]",
                ha='center', va='center', fontsize=12, color='#888888')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    pos = _topological_layout(nodes, edges)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    x_min, x_max = min(all_x)-2.2, max(all_x)+2.2
    y_min, y_max = min(all_y)-1.8, max(all_y)+1.6

    fig_w = max(8,  (x_max - x_min) * 1.05)
    fig_h = max(5,  (y_max - y_min) * 1.05 + (0.7 if title else 0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')
    ax.set_aspect('equal')

    if title:
        ax.text((x_min+x_max)/2, y_max-0.5, title,
                ha='center', va='top', fontsize=11, fontweight='bold',
                color='#2E4057')

    # Detect back-edges
    back_edges = set()
    for (src, dst, lbl) in edges:
        if src in pos and dst in pos:
            if pos[dst][1] >= pos[src][1]:
                back_edges.add((src, dst))

    # Draw edges under nodes
    for (src, dst, lbl) in edges:
        if src not in pos or dst not in pos:
            continue
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        _draw_arrow(ax, x1, y1, x2, y2, label=lbl,
                    bend=(src, dst) in back_edges)

    # Draw nodes
    for node_id, info in nodes.items():
        if node_id not in pos:
            continue
        x, y  = pos[node_id]
        shape = info['shape']
        label = info['label']
        if shape == 'rounded':
            _draw_pill(ax, x, y, 1.8, 0.65, label, C_ROUND, C_ROUND_TXT)
        elif shape == 'diamond':
            _draw_diamond(ax, x, y, 1.5, 0.95, label, C_DIA, C_DIA_TXT)
        else:
            _draw_rect(ax, x, y, 1.9, 0.72, label, C_RECT, C_RECT_TXT)

    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()