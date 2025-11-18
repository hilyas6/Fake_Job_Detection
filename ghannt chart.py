import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np

# --- Project Tasks ---
tasks = [
    {"Task": "1. Literature Review and Project Proposal", "Start": "2025-10-01", "End": "2025-11-10"},
    {"Task": "2. Dataset Preparation and Preprocessing", "Start": "2025-10-10", "End": "2025-10-20"},
    {"Task": "3. Model Selection and Initial Setup", "Start": "2025-10-20", "End": "2025-11-08"},
    {"Task": "4. Model Training and Optimization", "Start": "2025-10-20", "End": "2026-03-01"},
    {"Task": "5. Initial Project Report Preparation", "Start": "2025-11-15", "End": "2026-01-12"},
    {"Task": "6. Hybrid Integration and Explainability", "Start": "2026-01-15", "End": "2026-03-05"},
    {"Task": "7. Cross-Domain Evaluation and Testing", "Start": "2026-03-05", "End": "2026-03-15"},
    {"Task": "8. Prototype Deployment", "Start": "2026-03-10", "End": "2026-03-29"},
    {"Task": "9. Final Model Refinement", "Start": "2026-04-01", "End": "2026-04-15"},
    {"Task": "10. Final Report Preparation and Submission", "Start": "2026-01-12", "End": "2026-04-19"},
]

# Convert to datetime
for t in tasks:
    t["Start"] = datetime.strptime(t["Start"], "%Y-%m-%d")
    t["End"] = datetime.strptime(t["End"], "%Y-%m-%d")

# --- Milestones ---
milestones = [
    {"Date": "2025-11-10", "Label": "Proposal Due"},
    {"Date": "2026-01-12", "Label": "Contextual Report Due"},
    {"Date": "2026-04-20", "Label": "Final Submission"},
]
for m in milestones:
    m["Date"] = datetime.strptime(m["Date"], "%Y-%m-%d")

# Timeline anchor
start_global = datetime(2025, 10, 1)

def month_index(date):
    return (date.year - start_global.year) * 12 + (date.month - start_global.month) + (date.day / 30)

# Normalize task times
for t in tasks:
    t["StartNorm"] = month_index(t["Start"])
    t["EndNorm"] = month_index(t["End"])

# Normalize milestone times
for m in milestones:
    m["Norm"] = month_index(m["Date"])

# Colors
colors = ["#6D8BC7", "#5CB1CF", "#5BC4C9", "#60D5B5", "#85E89F",
          "#E28AB3", "#DE6E9D", "#E7A96B", "#E9E985", "#9EE29E"]

fig, ax = plt.subplots(figsize=(12, 6))

# Draw tasks
for i, task in enumerate(tasks):
    ax.barh(i,
            task["EndNorm"] - task["StartNorm"],
            left=task["StartNorm"],
            height=0.45,
            color=colors[i % len(colors)], edgecolor='black', linewidth=0.6)

ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t["Task"] for t in tasks], fontsize=9)
ax.invert_yaxis()

# ---- PERFECT MONTH GRID ----
num_months = 7  # Oct 2025 → Apr 2026
months = np.arange(0, num_months + 1, 1)
quarters = np.arange(0, num_months, 1/4)

ax.set_xticks(months)
ax.set_xticks(quarters, minor=True)

month_labels = pd.date_range("2025-10-01", "2026-05-01", freq="MS").strftime("%b '%y")
ax.set_xticklabels(month_labels)

ax.grid(which="major", axis="x", linestyle="--", alpha=0.65)
ax.grid(which="minor", axis="x", linestyle=":", alpha=0.35)
ax.grid(which="major", axis="y", linestyle=":", alpha=0.1)

# ---- ADD MILESTONES BACK (Correctly Positioned) ----
for m in milestones:
    ax.axvline(m["Norm"], color='#4B0082', linestyle='--', linewidth=1.4)
    ax.text(m["Norm"], -1.2, f"{m['Label']}\n({m['Date'].strftime('%d %b %y')})",
            ha='center', va='bottom', fontsize=9, color='#4B0082', fontweight='bold')

ax.set_xlabel("Timeline", fontsize=11)

plt.tight_layout()
plt.show()
