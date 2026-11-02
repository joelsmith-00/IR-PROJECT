"""Embedding models. Sentence-Transformers when installed (downloads a small model once); otherwise an
offline LSA embedder so the pipeline always runs."""
import numpy as np
from .retrieval import TfidfSpace, LsaSpace


class LsaEmbedder:
    name = "LSA-32 (offline fallback)"

    def __init__(self, docs, dim=32):
        self.l = LsaSpace(TfidfSpace(docs), dim)

    def encode_docs(self):
        return self.l.Z.astype(np.float32)

    def encode_query(self, text):
        return self.l.query_vec(text).astype(np.float32)


class SbertEmbedder:
    def __init__(self, docs, model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m, self.name, self.texts = SentenceTransformer(model), model, [d["text"] for d in docs]

    def encode_docs(self):
        return self.m.encode(self.texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def encode_query(self, text):
        return self.m.encode([text], normalize_embeddings=True)[0].astype(np.float32)


def get_embedder(docs, kind="auto"):
    """kind: 'auto' (sbert if available, else LSA), 'sbert', 'lsa'."""
    if kind in ("auto", "sbert"):
        try:
            return SbertEmbedder(docs)
        except Exception:
            if kind == "sbert":
                raise
    return LsaEmbedder(docs)
