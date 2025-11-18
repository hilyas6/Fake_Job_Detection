# main.py
import os
import torch
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from src import config
from src.dataset_loader import load_dataset, build_doc_features
from src.build_graph import build_graph
from src.model import TextGCN
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    # ---------------- DEVICE SETUP ---------------- #
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # ---------------- DATA LOADING ---------------- #
    print("📂 Loading dataset...")
    train_df, val_df, test_df = load_dataset(config.DATA_PATH)

    # ---------------- TF-IDF VOCAB ---------------- #
    print("🧠 Building TF-IDF vocabulary...")
    tfidf = TfidfVectorizer(
        max_features=config.MAX_TFIDF_FEATURES,
        min_df=config.MIN_WORD_FREQ,
        max_df=0.8,
    )
    tfidf.fit(train_df["text"])
    word2id = {w: i for i, w in enumerate(tfidf.vocabulary_.keys())}

    # ---------------- FEATURES ---------------- #
    print("🧩 Creating document features (BERT + TF-IDF + rules)...")
    x_docs = build_doc_features(train_df)
    y = torch.tensor(train_df["label"].values, dtype=torch.long)

    # ---------------- GRAPH ---------------- #
    print("🔗 Building graph...")
    edge_index, edge_weight = build_graph(train_df, word2id, tfidf)

    num_docs = len(train_df)
    num_words = len(word2id)
    feat_dim = x_docs.size(1)

    # Add zero-init word features
    x_words = torch.zeros((num_words, feat_dim), dtype=torch.float32)
    x = torch.cat([x_docs, x_words], dim=0)

    # ---------------- DATA OBJECT ---------------- #
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)

    # Fix masks: only documents are labeled
    data.train_mask = torch.zeros(num_docs + num_words, dtype=torch.bool)
    data.train_mask[:num_docs] = True
    data.val_mask = torch.zeros_like(data.train_mask)
    data.val_mask[:num_docs] = True
    data.test_mask = torch.zeros_like(data.train_mask)
    data.test_mask[:num_docs] = True

    data = data.to(device)

    print(f"✅ Graph built: {num_docs} docs, {num_words} words, "
          f"{edge_index.size(1)} edges, feature dim {feat_dim}")

    # ---------------- MODEL ---------------- #
    model = TextGCN(
        in_dim=feat_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=2,
        dropout=config.DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    class_weights = torch.tensor([1.0, 3.5], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ---------------- TRAIN ---------------- #
    print("🚀 Starting TextGCN training...")
    model = train_model(
        model,
        data,
        config.EPOCHS,
        optimizer,
        criterion,
        num_docs,
        config.MODEL_DIR,
        config.EARLY_STOP_PATIENCE,
        device=device
    )

    # ---------------- EVALUATE ---------------- #
    print("📊 Evaluating...")
    evaluate_model(model, data, num_docs, config.RESULT_DIR, target_recall=None, fixed_threshold=0.65)


if __name__ == "__main__":
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    main()
