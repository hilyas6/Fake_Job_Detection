
import torch
import numpy as np

def build_graph(df, word2id, tfidf_vectorizer):
    print(" Building fast TF-IDF-based graph...")
    doc_nodes = len(df)
    X = tfidf_vectorizer.transform(df["text"])
    coo = X.tocoo()

    # doc-word edges
    edges = np.vstack([coo.row, coo.col + doc_nodes])
    weights = coo.data.astype(np.float32)

    # add reverse edges (word->doc)
    rev_edges = np.vstack([edges[1], edges[0]])
    edges = np.hstack([edges, rev_edges])
    weights = np.hstack([weights, weights])

    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    print(f" Graph edges: {edge_index.shape[1]} (doc-word + word-doc)")
    return edge_index, edge_weight
