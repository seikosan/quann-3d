"""Keep sibling research projects' bare local imports isolated during collection."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def pytest_pycollect_makemodule(module_path, parent):
    path = Path(module_path).resolve()
    if "tests" not in path.parts:
        return None
    project = path.parents[1]
    project_src = project / "src"
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        source = Path(filename).resolve()
        if source == project_src or project_src in source.parents:
            continue
        if any(
            sibling != project and (sibling / "src" == source or (sibling / "src") in source.parents)
            for sibling in ROOT.iterdir()
            if sibling.is_dir()
        ):
            sys.modules.pop(name, None)
    return None
