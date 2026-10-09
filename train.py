import csv
import json
from pathlib import Path
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from router import ROOT, MODEL_PATH


def train():
    with (ROOT / "data" / "tickets.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    assert {r["group_id"] for r in train_rows}.isdisjoint({r["group_id"] for r in test_rows})
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                      ("classifier", LogisticRegression(C=4.0, max_iter=1000, random_state=42))])
    model.fit([r["text"] for r in train_rows], [r["category"] for r in train_rows])
    truth = [r["category"] for r in test_rows]
    predictions = model.predict([r["text"] for r in test_rows])
    labels = list(model.classes_)
    metrics = {"dataset": "AI-authored synthetic demo; not real customer tickets", "split": "Base issue groups held out; all variants of an issue stay together", "train_rows": len(train_rows), "test_rows": len(test_rows), "train_groups": len({r["group_id"] for r in train_rows}), "test_groups": len({r["group_id"] for r in test_rows}), "sklearn_version": sklearn.__version__, "accuracy": accuracy_score(truth, predictions), "macro_f1": f1_score(truth, predictions, average="macro"), "labels": labels, "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(), "classification_report": classification_report(truth, predictions, output_dict=True, zero_division=0), "errors": [{"text": r["text"], "expected": r["category"], "predicted": str(pred)} for r, pred in zip(test_rows, predictions) if r["category"] != pred]}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Synthetic held-out accuracy: {metrics['accuracy']:.1%}; macro F1: {metrics['macro_f1']:.3f}")
    print("Saved model and reports/evaluation.json. These are demo results, not production benchmarks.")
    return model, metrics

if __name__ == "__main__":
    train()
