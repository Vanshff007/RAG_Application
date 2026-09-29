import os

_model = None


def local_embed(texts, input_type):
    """Embeds texts on this machine with fastembed, so no API key is needed."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(
            os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            cache_dir=os.getenv("EMBED_CACHE_DIR") or None,
        )
    embed = _model.query_embed if input_type == "query" else _model.passage_embed
    return [vector.tolist() for vector in embed(texts)]
