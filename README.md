# TicketRoute — AI Support Ticket Classifier

A local machine learning application that categorizes a support message and suggests a team. Prepared for Adwaith Nambiar with ChatGPT assistance. This is an educational personal project, not a system used by a business.

## Features

- Six categories: billing, account, bug, feature request, shipping, and how-to.
- TF-IDF word unigrams/bigrams plus logistic regression.
- Browser interface, JSON prediction API, and health endpoint.
- Model score breakdown and a human-review fallback below 0.45.
- Reproducible training, held-out evaluation, confusion matrix, and automated tests.
- No paid API, GPU, or downloaded language model required.

## Windows CMD setup

Install Python 3.11 or 3.12, extract this folder, and open CMD in it.

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train.py
python -m unittest discover -s tests -v
python app.py
```

Use `py -3.12` instead if you have Python 3.12. Open http://127.0.0.1:5003. Stop with Ctrl+C. After initial setup, `start_windows.bat` trains and starts the app using the existing environment.

macOS/Linux: create an environment with `python3 -m venv .venv`, activate using `source .venv/bin/activate`, then run the same pip, training, testing, and app commands.

## How it works

Training fits the vectorizer on training text only, then learns a logistic regression classifier. Inference transforms a new ticket using that same fitted vectorizer. The largest predicted probability determines the suggested category; scores below 0.45 are routed to a person. The threshold is a demonstration heuristic, not validated against business costs. Scores are not calibrated certainty, and unknown or mixed-topic messages can still receive incorrect high scores.

## Data and evaluation

`data/tickets.csv` contains 180 AI-authored synthetic examples: 60 base issues with three wording variants each. There are 144 training rows from 48 issue groups and 36 held-out rows from 12 issue groups. Variants of one base issue never cross the split. This reduces direct variant leakage, but the small, stylistically similar synthetic dataset cannot establish real-world performance.

See `reports/evaluation.json` for measured accuracy, macro F1, per-class metrics, confusion matrix, and errors. The script records its scikit-learn version; compatible dependency versions can produce slightly different results. Do not treat this small test set as a tuning set. Future evaluation should use independent real tickets, broader labels, and human-reviewed examples.

Retrain with `python train.py` after editing data. Save the same model pipeline for inference to keep feature transformations consistent. The generated `models/router.joblib` is excluded from Git; train locally after cloning. Only load joblib files you trust.

## API

- `GET /health`: model readiness.
- `POST /api/predict`: JSON object with `text` (1–5000 characters).

```bat
curl -X POST http://127.0.0.1:5003/api/predict -H "Content-Type: application/json" -d "{\"text\":\"I was billed twice\"}"
```

The response includes `category`, `team`, `score`, `routing`, `needs_review`, and all category scores. Invalid inputs return HTTP 400; oversized requests return HTTP 413.

## Project files

| File | Purpose |
| --- | --- |
| `train.py` | Fit pipeline, evaluate held-out issues, save model |
| `router.py` | Load model, predict categories, apply routing threshold |
| `app.py` | Local Flask UI and REST API |
| `templates/index.html` | Responsive browser interface |
| `data/tickets.csv` | Synthetic labeled dataset with group split |
| `reports/evaluation.json` | Measured demo evaluation |
| `tests/test_app.py` | API validation, predictions, readiness, fallback tests |
| `start_windows.bat` | Start after environment setup |

## Upload to GitHub from Windows CMD

Create an empty repository named **ai-ticket-router** on https://github.com/new under **adwaithnambiar2**. Leave README, license, and gitignore options unchecked, since this folder includes them.

From the extracted project folder:

```bat
git init
git add .
git status --short
git commit -m "Add AI support ticket classifier"
git branch -M main
git config --local credential.username adwaithnambiar2
git remote add origin https://adwaithnambiar2@github.com/adwaithnambiar2/ai-ticket-router.git
git push -u origin main
```

Sign in as **adwaithnambiar2** when prompted. Before committing, check that `.venv` and `models` are not staged. The synthetic CSV and evaluation report should be included. If Git requests your author identity, set `git config --local user.name "Adwaith Nambiar"` and `git config --local user.email "YOUR_GITHUB_EMAIL"`, then retry the commit.

## Troubleshooting

- **Model missing:** run `python train.py` from the project folder.
- **Module missing:** activate `.venv` and install requirements with its Python.
- **Port occupied:** stop the other app or change port 5003 in `app.py`.
- **Incorrect prediction:** expected for a small synthetic dataset. Collect better data; keep a human review path.

## Next steps

Add independently collected examples, evaluate out-of-domain messages, validate probability calibration and thresholds, compare a baseline, and measure per-class errors. For deployment, use a production WSGI server and add authentication, monitoring, and a documented privacy policy. The included Flask server binds to localhost and is for local development.

## Attribution

Created with substantial ChatGPT assistance, including code and synthetic data. Review and understand the implementation before presenting it in interviews. This demonstrates local training and inference integration; uploading it to GitHub does not constitute production deployment.

MIT license; see LICENSE.
