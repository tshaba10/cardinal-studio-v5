"""Run with the project Python to regenerate diagrams; no API key needed."""

from pathlib import Path
from graphs import markdown, svg
from workflow import interview_graph, design_graph
from quality_review import quality_graph


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    graphs = {
        "Interview": interview_graph(),
        "Design, validation and review": design_graph(),
        "Optional dataset requirements review": quality_graph(),
    }
    (root / "langgraph_diagrams.md").write_text(markdown(graphs), encoding="utf-8")
    for name, graph in zip(("interview", "design", "quality"), graphs.values()):
        (root / (name + "_graph.svg")).write_text(svg(graph), encoding="utf-8")
    print("Created langgraph_diagrams.md and interview/design/quality graph SVGs.")
