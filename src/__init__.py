"""radiology-mini-vlm: CLIP-style retrieval + fairness audit (educational demo).

Lazy re-exports (PEP 562): importing this package never pulls in torch —
``src.retrieval`` (which needs torch/open_clip) is imported only when one of
its names is actually accessed.
"""

__all__ = [
    "embed_images",
    "embed_texts",
    "load_model",
    "rank_by_cosine",
    "text_to_image_search",
    "audit_csv",
    "make_demo_csv",
    "print_gap_table",
]

_LAZY_IMPORTS = {
    "embed_images": ".retrieval",
    "embed_texts": ".retrieval",
    "load_model": ".retrieval",
    "rank_by_cosine": ".retrieval",
    "text_to_image_search": ".retrieval",
    "audit_csv": ".fairness_audit",
    "make_demo_csv": ".fairness_audit",
    "print_gap_table": ".fairness_audit",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
