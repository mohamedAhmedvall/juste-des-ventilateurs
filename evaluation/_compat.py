"""Compatibilite Windows — forcer UTF-8 sur stdout/stderr.

Importer en tete de chaque script point d'entree pour eviter
UnicodeEncodeError sur CMD Windows (chcp 65001 ne suffit pas toujours).

Usage :
    import evaluation._compat  # noqa: F401  (side-effect only)
"""
import io
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
