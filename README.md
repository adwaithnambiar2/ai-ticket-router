# TicketRoute
### AI Support Ticket Classification & Routing

Turn a support message into a category, a suggested team, and a routing recommendation.

**TicketRoute** is a local machine learning application that uses **TF-IDF and logistic regression** to classify customer-support messages. A Flask API serves predictions to a responsive browser interface, with a human-review fallback when the model's top score is low.

**Python · scikit-learn · Flask · NumPy · HTML/CSS/JavaScript**

[Quick start](#quick-start) · [Model results](#model-results) · [API](#api) · [Project structure](#project-structure)

---

## Overview

Support messages cover everything from login problems to billing questions. TicketRoute explores how a lightweight text classifier can assist with initial triage.

Enter a message such as:

> I was billed twice for my subscription.

The application predicts a category, suggests the responsible team, and displays scores for all six categories. It recommends human review when the highest score is below **0.45**.

This is an educational prototype using synthetic data. Routing is a displayed recommendation; the app does not send tickets to external teams.

## Features

- **Six-category classification** covering common support topics.
- **Interactive web interface** with example tickets and score breakdowns.
- **Local inference** without a paid API, GPU, or external language model.
- **Human-review fallback** for predictions with low model scores.
- **Reusable ML pipeline** that saves text preprocessing and the classifier together.
- **Evaluation report** with accuracy, macro F1, confusion matrix, and misclassified examples.
- **Automated tests** for predictions, input validation, app readiness, and review routing.

### Categories and teams

| Category | Typical issue | Suggested team |
| --- | --- | --- |
| Account | Login, password, or verification problems | Account support |
| Billing | Payments, invoices, or refunds | Finance |
| Bug | Crashes, errors, or broken functionality | Engineering |
| Feature request | New capabilities or integrations | Product |
| Shipping | Deliveries, tracking, or damaged parcels | Fulfillment |
| How-to | Instructions for existing features | Customer success |

## How it works

### Training

1. Load labeled messages from `data/tickets.csv`.
2. Separate training and test data using predefined issue groups.
3. Fit a TF-IDF vectorizer on training text, using individual words and two-word phrases.
4. Train a multiclass logistic regression classifier.
5. Evaluate on held-out issue groups and save the pipeline and evaluation report.

### Prediction

The saved pipeline transforms a new message and returns category probabilities. The highest-scoring category determines the suggested team. If its score is below 0.45, the displayed routing becomes **Human review**.

**Design choice:** TF-IDF and logistic regression provide a compact baseline that is quick to train and easy to run on a CPU. They also expose the limitations of vocabulary-based classification when unfamiliar wording appears.

Scores are model estimates, not calibrated certainty. The review threshold is a demonstration heuristic; a confident prediction can still be incorrect.

## Model results

Results from the committed [evaluation report](reports/evaluation.json), produced with scikit-learn **1.6.1**:

| Metric | Result |
| --- | --- |
| Held-out accuracy | **75.0%** |
| Macro F1 | **0.717** |
| Training examples | 144 |
| Test examples | 36 |
| Training issue groups | 48 |
| Test issue groups | 12 |

### What the results show

The model correctly classified **27 of 36** test messages. It struggled with shipping messages and confused a refund delay with a delivery issue. Shipping recall was **0%** on this small held-out set, illustrating why overall accuracy alone is insufficient.

The dataset contains **180 AI-authored synthetic messages**: 60 base issues, each with three wording variants. All variants of an issue remain in the same split, preventing those closely related examples from appearing in both training and testing.

These results describe a small synthetic demonstration, not expected performance on real customer tickets. The test rows represent only 12 distinct base issues, so the 36 observations are not independent examples of customer behavior.

## Quick start

### Requirements

- Python **3.9–3.12**; Windows setup was verified with Python 3.9.
- Git, if cloning the repository.
- Internet access to install dependencies. Classification itself runs locally.

### 1. Clone the project

```bash
git clone https://github.com/adwaithnambiar2/ai-ticket-router.git
cd ai-ticket-router
```

### 2. Create a virtual environment

**Windows CMD**

```bat
py -3.9 -m venv .venv
.venv\Scripts\activate
```

If you have Python 3.11 or 3.12, substitute that version in the first command.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install and train

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
```

Training generates:

- `models/router.joblib` — fitted vectorizer and classifier.
- `reports/evaluation.json` — metrics, confusion matrix, and prediction errors.

The model file is excluded from Git and must be generated after cloning. Dependency resolution varies by Python version, so exact metrics may differ slightly between environments.

### 4. Run the tests

```bash
python -m unittest discover -s tests -v
```

The suite contains five tests covering API predictions, invalid input, page and health responses, probability totals, and uncertain-input review. Running it also retrains the model and regenerates the evaluation report.

### 5. Launch the application

```bash
python app.py
```

Open **http://127.0.0.1:5003**.

Keep the terminal open while using the app. Press **Ctrl+C** to stop the server. On Windows, `start_windows.bat` can train and launch the app once the environment is configured.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Confirm the application and model are loaded |
| POST | `/api/predict` | Classify a ticket message |

### Prediction request

```json
{
  "text": "I was billed twice for my subscription"
}
```

**Windows CMD example**

```bat
curl -X POST http://127.0.0.1:5003/api/predict -H "Content-Type: application/json" -d "{\"text\":\"I was billed twice for my subscription\"}"
```

### Response fields

| Field | Meaning |
| --- | --- |
| `category` | Highest-scoring category |
| `team` | Team associated with that category |
| `score` | Top model probability, rounded to four decimal places |
| `needs_review` | Whether the top score is below 0.45 |
| `routing` | Suggested team or Human review |
| `alternatives` | All category scores, ranked highest first |

Messages must contain non-whitespace text and be no longer than **5,000 characters**. Invalid inputs return **HTTP 400**; request bodies exceeding the configured limit return **HTTP 413**.

## Project structure

| File | Responsibility |
| --- | --- |
| `app.py` | Flask routes, validation, and local server |
| `router.py` | Model loading, prediction, and team routing |
| `train.py` | Training pipeline, evaluation, and model persistence |
| `templates/index.html` | Responsive interface and browser API requests |
| `data/tickets.csv` | Synthetic examples, labels, and split groups |
| `reports/evaluation.json` | Reproducible evaluation output |
| `tests/test_app.py` | Automated application and prediction tests |
| `requirements.txt` | Python dependency constraints |
| `start_windows.bat` | Windows launch helper |
| `.gitignore` | Excludes environments, generated models, and local secrets |
| `LICENSE` | MIT license |

## Limitations

- Synthetic examples provide limited language diversity and coverage.
- The classifier always chooses among its six labels; it has no trained unknown-topic class.
- Mixed-topic and unfamiliar messages may produce incorrect, high-scoring predictions.
- The review threshold has not been tuned against real operational costs.
- The Flask server runs on localhost for development. Authentication, monitoring, and production hosting are outside the current implementation.
- Joblib files should only be loaded from trusted sources.

## Future improvements

- Collect independently labeled real-world examples.
- Compare against a majority-class baseline and alternative text representations.
- Evaluate unknown-topic and mixed-topic messages.
- Calibrate probabilities and validate review thresholds on a separate validation set.
- Improve shipping and billing coverage without tuning on the held-out test examples.
- Add production serving, monitoring, and versioned model releases.

## Troubleshooting

| Issue | Fix |
| --- | --- |
| Browser says connection refused | Run `python app.py` and keep that terminal open |
| Model file is missing | Run `python train.py` before launching |
| A package cannot be imported | Activate `.venv` and install requirements using `python -m pip` |
| Python version is unavailable | On Windows, run `py -0p` and choose an installed supported version |
| Port 5003 is occupied | Stop the other server or change the port in `app.py` |
| A ticket is classified incorrectly | Review the scores and evaluation errors; this prototype requires better data for broader use |

## Author and attribution

**Adwaith Nambiar** · [GitHub](https://github.com/adwaithnambiar2)

Created with substantial ChatGPT assistance for code, documentation, and synthetic data. The project demonstrates a local workflow from training and evaluation to browser-based inference.

## License

Released under the [MIT License](LICENSE).
