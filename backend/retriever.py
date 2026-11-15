"""Multi-document retrieval.

A plain "top-k over everything" search can return all its results from one document and
miss the others. Here every source is searched on its own, and the merge step takes the best
passage from each source first, so every relevant document is represented in the answer.
"""
from typing import List, Optional

from backend.models import Evidence
from backend.vector_store import VectorStore
from config import settings
from utils.error_handler import EmbeddingError




def _to_evidence(hit) -> Evidence:
    meta = hit["metadata"]
    start = meta.get("video_timestamp", -1)
    end = meta.get("end_timestamp", -1)
    return Evidence(
        label="",
        text=hit["text"],
        distance=hit["distance"],
        document_id=meta["document_id"],
        filename=meta["filename"],
        source_type=meta["source_type"],
        chunk_id=meta["chunk_id"],
        page=meta["page_number"] or None,
        start=start if start >= 0 else None,
        end=end if end >= 0 else None,
        video_id=meta.get("video_id") or None,
    )


def retrieve(
    store: VectorStore,
    question: str,
    source_ids: List[str],
    top_k: Optional[int] = None,
    max_distance: Optional[float] = None,
) -> List[Evidence]:
    """Return up to top_k labelled passages (S1, S2, ...) from the given sources, best first.

    Returns [] when nothing is close enough, meaning the sources cannot answer the question.
    """
    top_k = settings.TOP_K if top_k is None else int(top_k)      # the default is used only when no value is given
    top_k = max(1, top_k)
    max_distance = settings.MAX_DISTANCE if max_distance is None else max_distance
    if not source_ids or not (question or "").strip():
        return []

    try:
        query_embedding = store.embedder.embed_one(question)
    except EmbeddingError:
        raise

    per_doc = []
    for document_id in source_ids:
        hits = store.query_source(query_embedding, document_id, top_k)
        if hits:
            per_doc.append(sorted(hits, key=lambda h: h["distance"]))

    candidates = [h for hits in per_doc for h in hits]
    if not candidates:
        return []
    best = min(h["distance"] for h in candidates)
    if best > max_distance:
        return []
    cutoff = min(max_distance, best + settings.RELATIVE_MARGIN)
    per_doc = [[h for h in hits if h["distance"] <= cutoff] for hits in per_doc]
    per_doc = [hits for hits in per_doc if hits]

    # round-robin: rank 0 from every source, then rank 1 from every source, ...
    per_doc.sort(key=lambda hits: hits[0]["distance"])
    chosen = []
    for rank in range(top_k):
        for hits in per_doc:
            if rank < len(hits) and len(chosen) < top_k:
                chosen.append(hits[rank])
    chosen.sort(key=lambda h: h["distance"])

    evidence = [_to_evidence(h) for h in chosen]
    for number, item in enumerate(evidence, start=1):
        item.label = f"S{number}"
    return evidence
