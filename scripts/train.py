from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_data
from src.modeling import train_all

if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data):,} sessions; synthetic demo data={data.attrs.get('synthetic', False)}")
    model, metrics, comparison = train_all(data)
    print("\nModel comparison (sorted by PR-AUC):")
    print(comparison[["model", "pr_auc", "roc_auc", "precision", "recall", "f1"]].to_string(index=False))
    print(f"\nSelected: {metrics['best_model']}")
    print("Artifacts saved in artifacts/.")
