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

## 🧪 Example CLI Usage

```bash
./vec-eyes \
  --dataset-list-hot /data/attacks \
  --dataset-list-cold /data/normal \
  --label WEB_ATTACK \
  --classify-objects /files/input \
  --method KnnCosine \
  --k 5 \
  --nlp-opt FastText \
  --output-csv report.csv \
  --output-json report.json
```

---

## 📄 YAML Rules (Detailed Explanation)

Vec-Eyes supports a powerful YAML-based rule engine.

### 🔹 Fields

```yaml
method: KnnCosine
k: 5
p: 2.0

rules:
  - title: Suspicious URL
    description: Detects common spam keywords
    match_rule: "casino|bonus|free"
    score: 80
```

### 🧠 Field Explanation

| Field        | Description |
|-------------|------------|
| `method`     | Classification method (KnnCosine, KnnEuclidean, KnnManhattan, KnnMinkowski, Bayes) |
| `k`          | Required for KNN – number of neighbors |
| `p`          | Required only for Minkowski metric |
| `title`      | Rule name (used in logs and reports) |
| `description`| Optional human-readable explanation |
| `match_rule` | Pattern to match (regex or vectorscan rule) |
| `score`      | Score (0–100) added to classification |

---

## 📄 YAML Example 1 (Security / Spam)

```yaml
method: KnnCosine
k: 5

rules:
  - title: Spam Keywords
    description: Common spam indicators
    match_rule: "free|bonus|win|casino"
    score: 70

  - title: Suspicious IP
    match_rule: "192\.168\.1\.100"
    score: 100
```

---

## 📄 YAML Example 2 (Biological / Scientific)

```yaml
method: KnnEuclidean
k: 3

rules:
  - title: Virus Pattern
    description: Detect virus-related sequences
    match_rule: "rna|virus|mutation"
    score: 80

  - title: Bacteria Signature
    match_rule: "e.coli|bacteria"
    score: 60
```

---

## 🏷️ Supported Labels

SPAM, MALWARE, PHISHING, ANOMALY, WEB_ATTACK, FUZZING, FLOOD,FRAUD, BLOCK_LIST, RAW_DATA,  
VIRUS, HUMAN, ANIMAL, CANCER, FUNGUS, BACTERIA

---

## ⚡ Performance

- Rust-native 🦀
- ndarray + BLAS ready
- Rayon parallelism
- High-throughput design

---

## 🔧 Optional VectorScan Support

### Fedora
```bash
sudo dnf install boost-devel cmake gcc gcc-c++
```

### Debian / Ubuntu
```bash
sudo apt install libboost-all-dev cmake build-essential
```

```bash
cargo build --features vectorscan
```

---

## 🤝 Contributing

Vec-Eyes is designed to evolve into a unified behavior intelligence engine.

We welcome contributions in:
- ML improvements
- Performance tuning
- New datasets
- Security rules
- Biological classification extensions

---

## 👤 Author

Orangewarrior

---

## ⭐ Star the project if you like it!
