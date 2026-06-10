from indexing.record import IndexRecord
from indexing.store import IndexStore, load_index

__all__ = ["IndexRecord", "IndexStore", "build_index", "load_index"]


def __getattr__(name: str):
    if name == "build_index":
        from indexing.pipeline import build_index

        return build_index
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
