# Claims Management System

## 🏗️ Tech Stack

- **Frontend**: Next.js 15.4.5 with React 19
- **Backend**: Next.js API Routes
- **Database**: PostgreSQL with Docker
- **Language**: TypeScript
- **Styling**: CSS Modules

## 🚀 Quick Start

### Prerequisites
- Node.js
- Docker and Docker Compose

### 1. Clone and Install
```bash
npm install
```

### 2. Database Setup
```bash
# Start PostgreSQL in Docker
npm run db:start

# Seed with sample data
npm run db:reset
```

### 3. Start Development
```bash
npm run dev
```

## 📁 Project Structure

```
src/
├── app/                   # Next.js App Router
│   └── api/               # API endpoints
├── components/            # UI components
└── database/              # Database related code
    ├── client.ts          # PostgreSQL connection
    ├── queries.ts         # Database queries
    ├── seed.ts            # Database seeding
    └── seed-data.json     # Sample data
```

## 🛠️ Available Scripts

| Command            | Description                             |
|--------------------|-----------------------------------------|
| `npm run dev`      | Start development server with Turbopack |
| `npm run db:start` | Start PostgreSQL container              |
| `npm run db:stop`  | Stop PostgreSQL container               |
| `npm run db:reset` | Reset and seed database                 |



<!-------------------------------------------------------------------------------- -->

# README — ML Model Export & Versioning Pipeline

## Project Overview

This project implements a complete Machine Learning pipeline for text classification using Python, Scikit-learn, and GitHub Actions.

The pipeline:

* preprocesses and cleans text data,
* trains a machine learning model,
* evaluates model performance,
* exports trained artifacts,
* versions model releases with Git tags,
* automates the workflow using GitHub Actions.

The project is designed as an introduction to MLOps practices and automated ML workflows.

---

# Project Structure

```text
dev-ia-P12/
│
├── javascript/
│   ├── datasets/
│   │   └── dataset_sample.csv
│   │
│   └── scripts/
│   │   ├── export_model.py
│   │   └── test_model.py
│   │
│   ├── models/
│   │   ├── model.pkl
│   │   ├── vectorizer.pkl
│   │   ├── label_encoder.pkl
│   │   └── metrics.json
│   │
│   ├── .github/
│   │   └── workflows/
│           └── github-actions-ML.yml
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

* Python 3.11
* Scikit-learn
* Pandas
* GitHub Actions
* Pickle
* Logistic Regression
* TF-IDF Vectorization

---

# Machine Learning Pipeline

## 1. Data Preparation

The dataset is loaded from:

```text
javascript/datasets/dataset_sample.csv
```

The preprocessing includes:

* cleaning company names,
* normalizing text,
* removing missing values,
* cleaning and regrouping tags,
* creating a unified text column.

Example:

```python id="bb5uw1"
df_new["text"] = (
    df_new["Consumer Claim"].fillna("")
    + " "
    + df_new["Company_final"].fillna("")
).str.lower()
```

---

# 2. Label Encoding

Target labels are encoded using:

```python id="32i3lm"
LabelEncoder()
```

This converts text categories into numerical labels.

---

# 3. Train/Test Split

The dataset is split into:

* 80% training data
* 20% testing data

Using:

```python id="sh10vb"
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

# 4. Text Vectorization

Text data is transformed using TF-IDF:

```python id="k91eyq"
TfidfVectorizer(
    max_features=5000,
    max_df=0.9,
    min_df=2
)
```

This converts text into numerical vectors usable by the ML model.

---

# 5. Model Training

The project uses:

```python id="3x55pz"
LogisticRegression()
```

Training:

```python id="ybkl7f"
model.fit(X_train_vec, y_train)
```

---

# 6. Model Evaluation

The model performance is evaluated using:

* Accuracy
* Classification Report
* Precision
* Recall
* F1-score

Example:

```python id="m8m6gj"
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
```

Metrics are exported into:

```text
models/metrics.json
```

---

# 7. Exporting Models

The pipeline exports:

* trained model,
* TF-IDF vectorizer,
* label encoder,
* metrics.

Files generated:

```text
models/model.pkl
models/vectorizer.pkl
models/label_encoder.pkl
models/metrics.json
```

Export example:

```python id="w9rxij"
pickle.dump(model, f)
```

---

# 8. Testing Model Reloading

The project includes a validation script:

```text
javascript/scripts/test_model.py
```

This script:

* reloads exported files,
* transforms sample text,
* performs predictions,
* validates that the exported artifacts work correctly.

Example:

```python id="ydpx3d"
prediction = model.predict(X_new)
```

---

# Git Versioning Strategy

The project uses Git tags for model versioning.

Example:

```bash id="jlwmya"
git tag v1.0.0
git push origin v1.0.0
```

Each tag creates:

* a new trained model,
* a GitHub Release,
* versioned exported artifacts.

---

# GitHub Actions Workflow

Workflow file:

```text
.github/workflows/github-actions-ML.yml
```

---

## Workflow Trigger

The pipeline runs:

* manually (`workflow_dispatch`)
* automatically when pushing a Git tag:

```yaml id="8eqaf0"
push:
  tags:
    - "v*"
```

---

## Workflow Steps

The workflow automatically:

1. checks out the repository,
2. installs dependencies,
3. trains the ML model,
4. exports artifacts,
5. tests exported files,
6. creates a GitHub Release.

---

# GitHub Release Artifacts

Each release contains:

* `model.pkl`
* `vectorizer.pkl`
* `label_encoder.pkl`
* `metrics.json`

These files can be downloaded directly from the GitHub Releases section.

---

# Run the Project Locally

## Install dependencies

```bash id="6v5r5x"
pip install -r requirements.txt
```

---

## Train and export the model

```bash id="pjlwmr"
python javascript/scripts/export_model.py
```

---

## Test exported artifacts

```bash id="v76g0k"
python javascript/scripts/test_model.py
```

---

# Create a New Model Release

## 1. Commit changes

```bash id="l5hlzm"
git add .
git commit -m "New ML version"
git push
```

---

## 2. Create a Git tag

```bash id="q6q0fk"
git tag v1.0.0
git push origin v1.0.0
```

This automatically triggers GitHub Actions.

---

# Future Improvements

Possible next steps:

* FastAPI deployment
* React frontend integration
* Docker containerization
* Automated ML tests
* Continuous retraining
* Cloud deployment

---

# Good Practices

* Keep models versioned with tags.
* Avoid committing large datasets.
* Use `.gitignore` for temporary files.
* Test exported models before deployment.
* Store metrics for reproducibility.

---

# Author

AI Developer Training Project
Machine Learning & MLOps Practice Project
