import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

# Clean, academic TextGCN architecture figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")

def box(x, y, w, h, text, color):
    """Draw a rounded box with centered text."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.3",
            fc=color, ec="#333333", lw=1
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=10, fontweight="semibold"
    )

# --- Draw boxes ---
box(3.2, 10.5, 3.5, 1, "Input Job Postings\n(Title + Description)", "#FFD6A5")
box(3.2, 9.0, 3.5, 1, "Text Preprocessing\n(Cleaning + Tokenisation + Lemmatization)", "#FDFFB6")
box(3.2, 7.3, 3.5, 1.3, "Feature Extraction\n• BERT Embeddings\n• TF-IDF Weights\n• Scam Indicators", "#B9FBC0")
box(3.2, 5.6, 3.5, 1.2, "Graph Construction\n(Document ↔ Word via TF-IDF Edges)", "#A3C4F3")
box(3.2, 4.0, 3.5, 1.0, "TextGCN Model\n(2 Graph Convolution Layers)", "#90DBF4")
box(3.2, 2.6, 3.5, 1.0, "Classifier\n(Real / Fake)", "#FFAFCC")
box(3.2, 1.3, 3.5, 0.8, "Predicted Label", "#FFC8DD")

# --- Connectors (arrows) ---
def connect(y1, y2):
    con = ConnectionPatch(
        (5, y1), (5, y2),
        "data", "data",
        arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333333"
    )
    ax.add_artist(con)

connect(10.5, 9.95)
connect(9.0, 8.55)
connect(7.3, 6.8)
connect(5.6, 5.1)
connect(4.0, 3.6)
connect(2.6, 2.1)

# --- Figure title ---
plt.text(
    5, 11.8, "Figure 1: Proposed TextGCN Architecture",
    ha="center", va="center",
    fontsize=12, fontweight="bold"
)

plt.tight_layout()
plt.savefig("Figure1_TextGCN_Architecture_Clean.png", dpi=300, bbox_inches="tight")
plt.show()
