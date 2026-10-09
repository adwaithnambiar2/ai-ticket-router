from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "router.joblib"
TEAMS = {"billing": "Finance", "account": "Account support", "bug": "Engineering", "feature_request": "Product", "shipping": "Fulfillment", "how_to": "Customer success"}

def load_model(path=MODEL_PATH):
    # Load only model files you created locally; joblib is not safe for untrusted files.
    if not path.exists():
        raise FileNotFoundError("Model missing. Run python train.py first.")
    return joblib.load(path)

def predict(model, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a nonempty ticket message.")
    if len(text) > 5000:
        raise ValueError("Ticket must be 5000 characters or fewer.")
    probabilities = model.predict_proba([text.strip()])[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
    category, score = ranked[0]
    review = float(score) < 0.45
    return {"category": str(category), "team": TEAMS[str(category)], "score": round(float(score), 4), "needs_review": review,
            "routing": "Human review" if review else TEAMS[str(category)],
            "alternatives": [{"category": str(c), "score": round(float(s), 4)} for c, s in ranked]}
