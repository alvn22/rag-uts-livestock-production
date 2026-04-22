from sentence_transformers import SentenceTransformer

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model

def embed_texts(texts):
    model = get_embedding_model()
    return model.encode(texts, batch_size=32, show_progress_bar=True)

def embed_query(query):
    model = get_embedding_model()
    return model.encode([query])[0]