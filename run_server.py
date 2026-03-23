"""
Server launcher that loads hallucination_guard modules from .pyc files
(used when .py source files are missing but __pycache__ bytecode exists).
"""
import sys
import os
import importlib.util
import importlib.abc

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PY_VER = f"cpython-{sys.version_info.major}{sys.version_info.minor}"


class PycFinder(importlib.abc.MetaPathFinder):
    """Load hallucination_guard and tests packages from .pyc when no .py exists."""

    def find_spec(self, fullname, path, target=None):
        if not (fullname.startswith("hallucination_guard") or fullname.startswith("tests")):
            return None

        parts = fullname.split(".")
        modname = parts[-1]

        search_dirs = list(path) if path else [PROJECT_ROOT]

        for base in search_dirs:
            # Package: subdirectory with __pycache__/__init__.cpython-310.pyc
            pkg_dir = os.path.join(base, modname)
            init_pyc = os.path.join(pkg_dir, "__pycache__", f"__init__.{PY_VER}.pyc")
            if os.path.isdir(pkg_dir) and os.path.exists(init_pyc):
                spec = importlib.util.spec_from_file_location(
                    fullname,
                    init_pyc,
                    submodule_search_locations=[pkg_dir],
                )
                if spec:
                    return spec

            # Module: __pycache__/<modname>.cpython-310.pyc
            mod_pyc = os.path.join(base, "__pycache__", f"{modname}.{PY_VER}.pyc")
            if os.path.exists(mod_pyc):
                spec = importlib.util.spec_from_file_location(fullname, mod_pyc)
                if spec:
                    return spec

        return None


# Install before any project imports
sys.meta_path.insert(0, PycFinder())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hallucination_guard.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
