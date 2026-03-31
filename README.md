# Vec-Eyes 🔍🧠

> **High-performance behavior intelligence engine and CLI written in Rust**

Vec-Eyes is a powerful, extensible behavior analysis platform designed to classify, detect, and analyze patterns across text, logs, datasets, and system traces.

Built with performance, flexibility, and real-world use cases in mind, Vec-Eyes combines:
- 🧠 Machine Learning (KNN, Naive Bayes)
- 🔡 NLP Pipelines (Tokenization, TF-IDF, Embeddings)
- ⚡ Vector-based similarity (Word2Vec, FastText)
- 🔎 Rule-based detection (Regex / optional VectorScan)
- 📊 Hybrid scoring engine

---

## 🚀 Why Vec-Eyes?

Vec-Eyes is not just a spam classifier.

It is a **behavior intelligence engine** capable of detecting patterns across multiple domains:

### 🔐 Security & Fraud Detection
- Spam & phishing emails
- Web attacks (SQLi, XSS, fuzzing)
- Malware behavior
- Fraud patterns in logs and transactions

### 🧬 Biological & Scientific Classification
Vec-Eyes can also be used for:
- Virus pattern classification
- Human / biological data classification
- Bacteria & fungus identification (textual/log patterns)
- Bioinformatics-style sequence classification (adaptable pipelines)

### 📊 General Pattern Recognition
- Log anomaly detection
- Dataset classification
- Behavioral clustering

---

## ⚙️ Core Features

### 🧠 Machine Learning
- KNN (Cosine, Euclidean, Manhattan, Minkowski)
- Naive Bayes (Count, TF-IDF)

### 🔡 NLP Engine
- Tokenization & normalization
- TF-IDF vectorization
- Word2Vec (lightweight training)
- FastText-style embeddings (subword support)

### 🔎 Rule Engine
- Regex matcher (default, no dependencies)
- Optional high-performance engine (VectorScan)
- YAML-driven rules with scoring system

### 📊 Hybrid Scoring
Combine:
- ML probability
- Rule matches
- Custom weights

---

## 📂 Project Structure

```
vec-eyes-lib/
vec-eyes-cli/
```

---
# Vec-Eyes CLI 🚀

> High-performance behavior classification CLI powered by Vec-Eyes Core

Vec-Eyes CLI is a production-ready command-line interface built on top of **vec-eyes-lib**, designed for real-world workflows in:

- 🔐 Security (web attacks, phishing, malware)
- 💰 Fraud detection (financial transactions, risk scoring)
- 🧬 Biological classification (virus, bacteria, anomaly patterns)
- 📊 General behavior intelligence pipelines

---

# ⚡ Why Vec-Eyes CLI?

- YAML-first configuration (reproducible pipelines)
- Multi-model ML engine (KNN, Bayes, SVM, RF, Boosting, IsolationForest)
- Hybrid scoring (ML + rule engine)
- Parallel execution via Rayon (`threads`)
- Designed for **real datasets**, not toy examples

---

# 🚀 Quick Start

```bash
cargo run --   --rules-yaml rules.yaml   --classify-objects ./samples/
```

---

# 🧪 Validate YAML

```bash
cargo run -- --validate-yaml rules.yaml
```

✔ Validates:
- required parameters
- model-specific constraints
- dataset paths

---

# 📄 Example 1 — Spam Detection (KNN + FastText)

```yaml
method: KnnCosine
nlp: FastText
k: 5
threads: 4

datasets:
  hot:
    - /data/email/spam/
  cold:
    - /data/email/normal/

rules:
  - title: Spam Keywords
    match_rule: "free|bonus|win|casino"
    score: 70
```

Run:

```bash
vec-eyes --rules-yaml spam.yaml --classify-objects ./emails/
```

---

# 📄 Example 2 — Web Attack Detection (RandomForest + OOB)

```yaml
method: RandomForest
nlp: FastText
threads: 8

random_forest_mode: ExtraTrees
random_forest_n_trees: 200
random_forest_bootstrap: true
random_forest_oob_score: true

datasets:
  hot:
    - /data/http/attacks/
  cold:
    - /data/http/normal/

rules:
  - title: SQL Injection
    match_rule: "union select|or 1=1"
    score: 90
```

---

# 📄 Example 3 — Fraud Detection (Logistic Regression)

```yaml
method: LogisticRegression
nlp: TfIdf
threads: 4

logistic_learning_rate: 0.01
logistic_epochs: 100

datasets:
  hot:
    - /data/fraud/high-risk/
  cold:
    - /data/fraud/low-risk/
```

---

# 📄 Example 4 — Anomaly Detection (Isolation Forest)

```yaml
method: IsolationForest
nlp: FastText

isolation_forest_n_trees: 150
isolation_forest_contamination: 0.02

datasets:
  hot:
    - /data/anomaly/outliers/
  cold:
    - /data/anomaly/normal/
```

---

# ⚙️ CLI Arguments Overview

| Argument | Description |
|--------|------------|
| `--rules-yaml` | Path to YAML config |
| `--validate-yaml` | Validate config only |
| `--classify-objects` | Directory of files to classify |
| `--threads` | Override thread count |
| `--output-json` | Export results as JSON |
| `--output-csv` | Export results as CSV |

---

# 🧠 Performance Notes

- `threads` controls Rayon parallelism
- KNN → parallel distance computation
- Bayes → parallel scoring
- RandomForest / Boosting → parallel training

---

# 🧠 Model Selection Guide

| Model | Best For |
|------|---------|
| KNN | Similarity / noisy text |
| Bayes | Fast baseline |
| Logistic | Fraud / production baseline |
| SVM | Text classification |
| RandomForest | Structured signals |
| IsolationForest | Anomaly detection |

---

# 🧩 Example (Advanced CLI Override)

```bash
vec-eyes   --rules-yaml rules.yaml   --threads 8   --random-forest-n-trees 300   --classify-objects ./traffic/
```

---

# 📊 Output

Example JSON output:

```json
{
  "file": "sample.txt",
  "classification": ["WEB_ATTACK"],
  "score": 92.5
}
```

---

# 🤝 Contributing

We welcome contributions:

- New datasets
- Performance improvements
- New classifiers
- Better YAML validation

---

# 💬 Final Note

Vec-Eyes CLI is not just a wrapper.

It is a **production-grade behavior intelligence interface** designed to bridge:
- ML pipelines
- rule-based detection
- real-world data workflows

Built in Rust. Designed for performance. Ready for serious use.


## 👤 Author

Orangewarrior

---

## ⭐ Star the project if you like it!
