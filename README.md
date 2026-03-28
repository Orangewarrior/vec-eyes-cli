# Vec-Eyes CLI

Vec-Eyes CLI is the operational face of the project. It turns the library into a practical workflow for classifying real files, recursive datasets, and rule-boosted detection pipelines.

## Why this CLI is useful

You can point Vec-Eyes at:

- mail buffers
- phishing samples
- HTTP attack requests
- anomaly traces
- malware-oriented notes
- Fraud classify
- biological labels and raw text corpora

and get a report back in CSV or JSON.

## Build

### Default build

```bash
cargo test
cargo run -- --help
```

### Native Vectorscan build

```bash
cargo test --features vectorscan
```

## System packages for native Vectorscan

### Fedora

```bash
sudo dnf install cmake gcc gcc-c++ boost-devel
```

### Debian / Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libboost-all-dev
```

## YAML-driven usage

```bash
./vec-eyes --yaml-rules /path/rules.yaml --classify-objects /path/to/files
```

If `--yaml-rules` is provided, the CLI loads method, NLP mode, recursion, dataset paths, scoring behavior, `k`, `p`, and output files from YAML.

## Direct CLI usage

```bash
./vec-eyes \
  --dataset-list-hot /data/http/attack/requests/ \
  --dataset-list-cold /data/http/regular/requests/ \
  --label WEB_ATTACK \
  --load-alert-ip ip/alert/spam_address_ip.txt \
  --load-alert-url url/alert_list/spam_url.txt \
  --classify-objects /files/classify \
  --threads 8 \
  --output-csv report_2026.csv \
  --output-json report.json \
  --method knn-cosine \
  --nlp-opt fast-text \
  --k 5 \
  --score-sum ON
```

## Minkowski example

```bash
./vec-eyes \
  --dataset-list-hot /data/attack \
  --dataset-list-cold /data/normal \
  --classify-objects /queue \
  --method knn-minkowski \
  --nlp-opt word2-vec \
  --k 7 \
  --p 3.0
```

## UX behavior

- running without arguments prints the long help output
- argument parsing errors print the clap help and examples
- the banner shows the current version and the author

## More documentation

See `../helper.md` for the YAML contract and the exact validation rules.
