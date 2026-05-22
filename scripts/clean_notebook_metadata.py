"""Remove widget metadata that can break GitHub notebook rendering.

Usage:
    python scripts/clean_notebook_metadata.py notebooks/00_agriculture_rag_from_scratch.ipynb
"""

import json
import sys
from pathlib import Path


def clean_notebook(path):
    notebook_path = Path(path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    metadata = notebook.setdefault("metadata", {})
    metadata.pop("widgets", None)

    for cell in notebook.get("cells", []):
        cell_metadata = cell.setdefault("metadata", {})
        cell_metadata.pop("widgets", None)

        for output in cell.get("outputs", []):
            data = output.get("data")
            if isinstance(data, dict):
                data.pop("application/vnd.jupyter.widget-view+json", None)

    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/clean_notebook_metadata.py <notebook.ipynb>")

    clean_notebook(sys.argv[1])
