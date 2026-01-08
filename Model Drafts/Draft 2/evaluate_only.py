# evaluate_only.py
import os
import torch
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from src import config
from src.dataset_loader import load_dataset, build_doc_features
from src.build_graph import build_graph
from src.model import TextGCN
from src.evaluate import evaluate_model


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # ---- Load dataset and rebuild graph structure ---- #
    print("📂 Loading dataset...")
    train_df, val_df, test_df = load_dataset(config.DATA_PATH)

    print("🧠 Building TF-IDF vocabulary...")
    tfidf = TfidfVectorizer(
        max_features=config.MAX_TFIDF_FEATURES,
        min_df=config.MIN_WORD_FREQ,
        max_df=0.8,
    )
    tfidf.fit(train_df["text"])
    word2id = {w: i for i, w in enumerate(tfidf.vocabulary_.keys())}

    # ---- Build features & graph (same as training) ---- #
    print("🧩 Creating document features...")
    x_docs = build_doc_features(train_df)
    y = torch.tensor(train_df["label"].values, dtype=torch.long)
    edge_index, edge_weight = build_graph(train_df, word2id, tfidf)

    num_docs = len(train_df)
    num_words = len(word2id)
    feat_dim = x_docs.size(1)
    x_words = torch.zeros((num_words, feat_dim), dtype=torch.float32)
    x = torch.cat([x_docs, x_words], dim=0)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    data.train_mask = torch.zeros(num_docs + num_words, dtype=torch.bool)
    data.train_mask[:num_docs] = True
    data.val_mask = torch.zeros_like(data.train_mask)
    data.val_mask[:num_docs] = True
    data.test_mask = torch.zeros_like(data.train_mask)
    data.test_mask[:num_docs] = True
    data = data.to(device)

    # ---- Load trained model ---- #
    model_path = os.path.join(config.MODEL_DIR, "textgcn_best.pt")
    if not os.path.exists(model_path):
        print("❌ Trained model not found! Run main.py first.")
        return

    model = TextGCN(
        in_dim=feat_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=2,
        dropout=config.DROPOUT,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"✅ Loaded trained model from {model_path}")

    # ---- Evaluate ---- #
    evaluate_model(model, data, num_docs, config.RESULT_DIR, fixed_threshold=0.65)


if __name__ == "__main__":
    main()
