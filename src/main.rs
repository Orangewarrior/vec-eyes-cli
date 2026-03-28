use clap::{CommandFactory, Parser, ValueEnum};
use rayon::ThreadPoolBuilder;
use std::path::PathBuf;
use vec_eyes_lib::classifier::{run_rules_pipeline, MethodKind};
use vec_eyes_lib::config::{ExtraMatchConfig, ExtraMatchEngine, RecursiveMode, RulesFile, ScoreSumMode};
use vec_eyes_lib::labels::ClassificationLabel;
use vec_eyes_lib::nlp::NlpOption;

const APP_VERSION: &str = "0.3.0";
const APP_AUTHOR: &str = "Orangewarrior";

#[derive(Debug, Clone, ValueEnum)]
enum MethodArg {
    Bayes,
    KnnCosine,
    KnnEuclidean,
    KnnManhattan,
    KnnMinkowski,
}

#[derive(Debug, Clone, ValueEnum)]
enum NlpArg {
    Count,
    TfIdf,
    Word2Vec,
    FastText,
}

#[derive(Debug, Parser)]
#[command(name = "vec-eyes")]
#[command(version = APP_VERSION)]
#[command(author = APP_AUTHOR)]
#[command(about = "Vec-Eyes is a behavior classification CLI for spam, phishing, web attacks, malware notes, anomaly traces, and raw text intelligence.")]
#[command(long_about = None)]
#[command(after_help = "Examples:
  vec-eyes --yaml-rules ./rules.yaml --classify-objects ./samples/mail
  vec-eyes --dataset-list-hot ./data/http/attack --dataset-list-cold ./data/http/regular --classify-objects ./incoming --method knn-cosine --nlp-opt fast-text --k 5 --threads 8 --output-csv report.csv --output-json report.json
  vec-eyes --dataset-list-hot ./data/attack --dataset-list-cold ./data/normal --classify-objects ./queue --method knn-minkowski --nlp-opt word2-vec --k 7 --p 3.0 --score-sum ON")]
struct Cli {
    #[arg(long)]
    yaml_rules: Option<PathBuf>,

    #[arg(long = "dataset-list-hot")]
    dataset_list_hot: Option<PathBuf>,

    #[arg(long = "dataset-list-cold")]
    dataset_list_cold: Option<PathBuf>,

    #[arg(long = "label")]
    hot_label: Option<String>,

    #[arg(long = "cold-label")]
    cold_label: Option<String>,

    #[arg(long = "load-alert-ip")]
    load_alert_ip: Option<PathBuf>,

    #[arg(long = "load-alert-url")]
    load_alert_url: Option<PathBuf>,

    #[arg(long = "classify-objects")]
    classify_objects: Option<PathBuf>,

    #[arg(long = "threads", default_value_t = 1)]
    threads: usize,

    #[arg(long = "output-csv")]
    output_csv: Option<PathBuf>,

    #[arg(long = "output-json")]
    output_json: Option<PathBuf>,

    #[arg(long = "method")]
    method: Option<MethodArg>,

    #[arg(long = "nlp-opt")]
    nlp_opt: Option<NlpArg>,

    #[arg(long = "score-sum", default_value = "OFF")]
    score_sum: String,

    #[arg(long = "k")]
    k: Option<usize>,

    #[arg(long = "p")]
    p: Option<f32>,
}

fn print_banner() {
    println!("Vec-Eyes v{} | author: {}", APP_VERSION, APP_AUTHOR);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args_os().len() == 1 {
        print_banner();
        let mut cmd = Cli::command();
        cmd.print_long_help()?;
        println!();
        return Err("no arguments provided".into());
    }

    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(err) => {
            print_banner();
            err.print()?;
            return Err("invalid CLI arguments".into());
        }
    };

    print_banner();

    if cli.threads > 1 {
        let _ = ThreadPoolBuilder::new().num_threads(cli.threads).build_global();
    }

    let rules = if let Some(yaml_rules) = &cli.yaml_rules {
        let content = std::fs::read_to_string(yaml_rules)?;
        let rules = serde_yaml::from_str::<RulesFile>(&content)?;
        rules.validate()?;
        rules
    } else {
        build_rules_from_cli(&cli)?
    };

    let classify_dir = cli
        .classify_objects
        .clone()
        .or_else(|| Some(rules.hot_test_path.clone()))
        .ok_or("missing classify path")?;

    let report = run_rules_pipeline(&rules, &classify_dir)?;

    if let Some(csv_path) = &rules.csv_output {
        report.write_csv(csv_path)?;
    }
    if let Some(json_path) = &rules.json_output {
        report.write_json(json_path)?;
    }

    for record in report.records {
        println!(
            "{} | {} | {} | {:.2}",
            record.title_object,
            record.name_file_dataset,
            record.classify_names_list,
            record.score_percent
        );
    }

    Ok(())
}

fn build_rules_from_cli(cli: &Cli) -> Result<RulesFile, Box<dyn std::error::Error>> {
    let method = match cli.method.as_ref().unwrap_or(&MethodArg::KnnCosine) {
        MethodArg::Bayes => MethodKind::Bayes,
        MethodArg::KnnCosine => MethodKind::KnnCosine,
        MethodArg::KnnEuclidean => MethodKind::KnnEuclidean,
        MethodArg::KnnManhattan => MethodKind::KnnManhattan,
        MethodArg::KnnMinkowski => MethodKind::KnnMinkowski,
    };

    let nlp = match cli.nlp_opt.as_ref().unwrap_or(&NlpArg::FastText) {
        NlpArg::Count => NlpOption::Count,
        NlpArg::TfIdf => NlpOption::TfIdf,
        NlpArg::Word2Vec => NlpOption::Word2Vec,
        NlpArg::FastText => NlpOption::FastText,
    };

    let hot_label = cli.hot_label
        .clone()
        .and_then(|s| s.parse::<ClassificationLabel>().ok())
        .unwrap_or(ClassificationLabel::WebAttack);

    let cold_label = cli.cold_label
        .clone()
        .and_then(|s| s.parse::<ClassificationLabel>().ok())
        .unwrap_or(ClassificationLabel::RawData);

    let score_sum = if cli.score_sum.eq_ignore_ascii_case("ON") {
        ScoreSumMode::On
    } else {
        ScoreSumMode::Off
    };

    if method.is_knn() {
        let k = cli.k.ok_or("KNN methods require --k <usize>")?;
        if k == 0 {
            return Err("KNN methods require --k >= 1".into());
        }
        if method.requires_p() {
            let p = cli.p.ok_or("KnnMinkowski requires --p <float>")?;
            if p <= 0.0 {
                return Err("KnnMinkowski requires --p > 0".into());
            }
        }
    }

    let mut extra_match = Vec::new();

    if let Some(path) = &cli.load_alert_ip {
        extra_match.push(ExtraMatchConfig {
            recursive_way: RecursiveMode::On,
            engine: ExtraMatchEngine::Regex,
            path: path.clone(),
            score_add_points: 15.0,
            title: Some("alert-ip".to_string()),
            description: Some("CLI loaded IP alert list".to_string()),
        });
    }

    if let Some(path) = &cli.load_alert_url {
        extra_match.push(ExtraMatchConfig {
            recursive_way: RecursiveMode::On,
            engine: ExtraMatchEngine::Regex,
            path: path.clone(),
            score_add_points: 15.0,
            title: Some("alert-url".to_string()),
            description: Some("CLI loaded URL alert list".to_string()),
        });
    }

    let rules = RulesFile {
        report_name: Some("Vec-Eyes CLI Report".to_string()),
        method,
        nlp,
        threads: Some(cli.threads),
        csv_output: cli.output_csv.clone(),
        json_output: cli.output_json.clone(),
        recursive_way: RecursiveMode::On,
        hot_test_path: cli.dataset_list_hot.clone().ok_or("missing --dataset-list-hot")?,
        cold_test_path: cli.dataset_list_cold.clone().ok_or("missing --dataset-list-cold")?,
        hot_label: Some(hot_label),
        cold_label: Some(cold_label),
        score_sum,
        extra_match,
        k: cli.k,
        p: cli.p,
    };

    rules.validate()?;
    Ok(rules)
}
