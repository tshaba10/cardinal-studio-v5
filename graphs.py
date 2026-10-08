"""Export real LangGraph topology as Mermaid and offline SVG, without a web renderer."""

from html import escape


def markdown(graphs):
    return "# Cardinal Studio — actual LangGraph diagrams\n\n" + "\n\n".join(
        f"## {name}\n\n```mermaid\n{graph.get_graph().draw_mermaid()}\n```"
        for name, graph in graphs.items()
    )


def svg(graph, active=None):
    data = graph.get_graph()
    # Small graph layout: deterministic order with START/END at the ends.
    nodes = sorted(
        data.nodes,
        key=lambda n: (
            0 if n == "__start__" else 2 if n == "__end__" else 1,
            {"designer": 0, "validator": 1, "reviewer": 2}.get(n, 0),
            n,
        ),
    )
    centers = {node: 65 + 100 * i for i, node in enumerate(nodes)}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="{len(nodes) * 100 + 30}" role="img" aria-label="LangGraph workflow">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#0e7490"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]
    for index, edge in enumerate(data.edges):
        a, b = centers[edge.source], centers[edge.target]
        dash = ' stroke-dasharray="6,4"' if edge.conditional else ""
        if b == a + 100:
            path = f"M 290 {a + 22} L 290 {b - 24}"
        else:
            x = 440 + (index % 3) * 45
            path = f"M 400 {a} C {x} {a}, {x} {b}, 400 {b}"
        parts.append(
            f'<path d="{path}" fill="none" stroke="#0e7490" stroke-width="2"{dash} marker-end="url(#arrow)"/>'
        )
        if edge.data:
            parts.append(
                f'<text x="430" y="{(a + b) / 2}" fill="#334155" font-size="13">{escape(str(edge.data))}</text>'
            )
    for node, y in centers.items():
        color = "#99f6e4" if node == active else "#e0f2fe"
        parts.append(
            f'<rect x="180" y="{y - 22}" width="220" height="44" rx="12" fill="{color}" stroke="#155e75"/>'
        )
        label = "dataset reviewer" if node == "dataset_reviewer" else node.strip("_")
        parts.append(
            f'<text x="290" y="{y + 5}" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#0f172a">{escape(label)}</text>'
        )
    return "".join(parts) + "</svg>"
