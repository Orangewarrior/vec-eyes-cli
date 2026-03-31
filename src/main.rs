
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use clap::{CommandFactory, Parser, ValueEnum};
use vec_eyes_lib::{
    collect_files_recursively, read_text_file, ClassificationLabel, ClassificationReport,
    ClassifierFactory, ExtraMatchConfig, ExtraMatchEngine, MethodKind, NlpOption,
    RandomForestMaxFeatures, RandomForestMode, RecursiveMode, RulesFile, ScoreSumMode,
    ScoringEngine, SvmKernel,
};

#[derive(Debug, Parser)]
#[command(
    name = "vec-eyes",
    version = "0.2.0",
    author = "Orangewarrior",
    about = "Vec-Eyes CLI: modular behavior classification, anomaly detection, and YAML-driven validation"
)]
#[command(after_help = EXAMPLES)]
struct Cli {
    /// Validate a YAML rules file and exit.
    #[arg(long)]
    validate_yaml: Option<PathBuf>,

    /// Load a YAML rules file, optionally override fields, and run classification.
    #[arg(long)]
    rules_yaml: Option<PathBuf>,

    /// Print the effective configuration as YAML after applying CLI overrides.
    #[arg(long, default_value_t = false)]
    print_effective_yaml: bool,

    /// Report name override.
    #[arg(long)]
    report_name: Option<String>,

    /// Target directory to classify recursively depending on recursive mode.
    #[arg(long)]
    classify_objects: Option<PathBuf>,

    /// Output CSV report path.
    #[arg(long)]
    output_csv: Option<PathBuf>,

    /// Output JSON report path.
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Classification method.
    #[arg(long, value_enum)]
    method: Option<MethodArg>,

    /// NLP option.
    #[arg(long, value_enum)]
    nlp: Option<NlpArg>,

    /// Number of threads used by parallel classifiers.
    #[arg(long)]
    threads: Option<usize>,

    /// Hot/positive training path.
    #[arg(long)]
    hot_path: Option<PathBuf>,

    /// Cold/negative training path.
    #[arg(long)]
    cold_path: Option<PathBuf>,

    /// Label used for hot data.
    #[arg(long, value_enum)]
    hot_label: Option<LabelArg>,

    /// Label used for cold data.
    #[arg(long, value_enum)]
    cold_label: Option<LabelArg>,

    /// Recursive mode for dataset loading and classification traversal.
    #[arg(long, value_enum)]
    recursive: Option<OnOffArg>,

    /// Score summing mode for ML + rule scoring.
    #[arg(long, value_enum)]
    score_sum: Option<OnOffArg>,

    /// Optional extra regex / vectorscan rule paths.
    #[arg(long)]
    extra_match_path: Vec<PathBuf>,

    /// Engine for extra-match paths. Reused for every extra-match path in this CLI flow.
    #[arg(long, value_enum)]
    extra_match_engine: Option<ExtraEngineArg>,

    /// Additional score added when an extra-match rule hits.
    #[arg(long)]
    extra_match_score_add_points: Option<f32>,

    /// KNN neighbors.
    #[arg(long)]
    k: Option<usize>,

    /// KNN Minkowski p parameter.
    #[arg(long)]
    p: Option<f32>,

    /// Logistic regression learning rate.
    #[arg(long)]
    logistic_learning_rate: Option<f32>,

    /// Logistic regression epochs.
    #[arg(long)]
    logistic_epochs: Option<usize>,

    /// Logistic regression lambda.
    #[arg(long)]
    logistic_lambda: Option<f32>,

    /// Random Forest mode.
    #[arg(long, value_enum)]
    random_forest_mode: Option<RandomForestModeArg>,

    /// Random Forest number of trees.
    #[arg(long)]
    random_forest_n_trees: Option<usize>,

    #[arg(long)]
    random_forest_max_depth: Option<usize>,

    #[arg(long, value_enum)]
    random_forest_max_features: Option<RandomForestMaxFeaturesArg>,

    #[arg(long)]
    random_forest_min_samples_split: Option<usize>,

    #[arg(long)]
    random_forest_min_samples_leaf: Option<usize>,

    #[arg(long)]
    random_forest_bootstrap: Option<bool>,

    #[arg(long)]
    random_forest_oob_score: Option<bool>,

    #[arg(long, value_enum)]
    svm_kernel: Option<SvmKernelArg>,

    #[arg(long)]
    svm_c: Option<f32>,

    #[arg(long)]
    svm_learning_rate: Option<f32>,

    #[arg(long)]
    svm_epochs: Option<usize>,

    #[arg(long)]
    svm_gamma: Option<f32>,

    #[arg(long)]
    svm_degree: Option<usize>,

    #[arg(long)]
    svm_coef0: Option<f32>,

    #[arg(long)]
    gradient_boosting_n_estimators: Option<usize>,

    #[arg(long)]
    gradient_boosting_learning_rate: Option<f32>,

    #[arg(long)]
    gradient_boosting_max_depth: Option<usize>,

    #[arg(long)]
    isolation_forest_n_trees: Option<usize>,

    #[arg(long)]
    isolation_forest_contamination: Option<f32>,

    #[arg(long)]
    isolation_forest_subsample_size: Option<usize>,
}

const EXAMPLES: &str = r#"
Examples:
  vec-eyes --validate-yaml rules.yaml
  vec-eyes --rules-yaml rules.yaml --classify-objects ./samples --output-json report.json
  vec-eyes --method KnnCosine --nlp FastText --k 5 --hot-path ./hot --cold-path ./cold --classify-objects ./samples
  vec-eyes --rules-yaml rules.yaml --threads 8 --random-forest-oob-score true --print-effective-yaml
"#;

#[derive(Clone, Debug, ValueEnum)]
enum MethodArg {
    Bayes,
    KnnCosine,
    KnnEuclidean,
    KnnManhattan,
    KnnMinkowski,
    LogisticRegression,
    RandomForest,
    IsolationForest,
    Svm,
    GradientBoosting,
}

impl From<MethodArg> for MethodKind {
    fn from(value: MethodArg) -> Self {
        match value {
            MethodArg::Bayes => MethodKind::Bayes,
            MethodArg::KnnCosine => MethodKind::KnnCosine,
            MethodArg::KnnEuclidean => MethodKind::KnnEuclidean,
            MethodArg::KnnManhattan => MethodKind::KnnManhattan,
            MethodArg::KnnMinkowski => MethodKind::KnnMinkowski,
            MethodArg::LogisticRegression => MethodKind::LogisticRegression,
            MethodArg::RandomForest => MethodKind::RandomForest,
            MethodArg::IsolationForest => MethodKind::IsolationForest,
            MethodArg::Svm => MethodKind::Svm,
            MethodArg::GradientBoosting => MethodKind::GradientBoosting,
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum NlpArg {
    Count,
    TfIdf,
    Word2Vec,
    FastText,
}
impl From<NlpArg> for NlpOption {
    fn from(value: NlpArg) -> Self {
        match value {
            NlpArg::Count => NlpOption::Count,
            NlpArg::TfIdf => NlpOption::TfIdf,
            NlpArg::Word2Vec => NlpOption::Word2Vec,
            NlpArg::FastText => NlpOption::FastText,
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum LabelArg { Spam, Malware, Phishing, Anomaly, Fuzzing, WebAttack, Flood, Porn, RawData, BlockList, Virus, Human, Animal, Cancer, Fungus, Bacteria, Free }
impl From<LabelArg> for ClassificationLabel {
    fn from(value: LabelArg) -> Self {
        match value {
            LabelArg::Spam => ClassificationLabel::Spam,
            LabelArg::Malware => ClassificationLabel::Malware,
            LabelArg::Phishing => ClassificationLabel::Phishing,
            LabelArg::Anomaly => ClassificationLabel::Anomaly,
            LabelArg::Fuzzing => ClassificationLabel::Fuzzing,
            LabelArg::WebAttack => ClassificationLabel::WebAttack,
            LabelArg::Flood => ClassificationLabel::Flood,
            LabelArg::Porn => ClassificationLabel::Porn,
            LabelArg::RawData => ClassificationLabel::RawData,
            LabelArg::BlockList => ClassificationLabel::BlockList,
            LabelArg::Virus => ClassificationLabel::Virus,
            LabelArg::Human => ClassificationLabel::Human,
            LabelArg::Animal => ClassificationLabel::Animal,
            LabelArg::Cancer => ClassificationLabel::Cancer,
            LabelArg::Fungus => ClassificationLabel::Fungus,
            LabelArg::Bacteria => ClassificationLabel::Bacteria,
            LabelArg::Free => ClassificationLabel::Free,
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum OnOffArg { On, Off }
impl From<OnOffArg> for RecursiveMode {
    fn from(value: OnOffArg) -> Self { match value { OnOffArg::On => RecursiveMode::On, OnOffArg::Off => RecursiveMode::Off } }
}
impl From<OnOffArg> for ScoreSumMode {
    fn from(value: OnOffArg) -> Self { match value { OnOffArg::On => ScoreSumMode::On, OnOffArg::Off => ScoreSumMode::Off } }
}

#[derive(Clone, Debug, ValueEnum)]
enum ExtraEngineArg { Regex, Vectorscan }
impl From<ExtraEngineArg> for ExtraMatchEngine {
    fn from(value: ExtraEngineArg) -> Self { match value { ExtraEngineArg::Regex => ExtraMatchEngine::Regex, ExtraEngineArg::Vectorscan => ExtraMatchEngine::Vectorscan } }
}

#[derive(Clone, Debug, ValueEnum)]
enum RandomForestModeArg { Standard, Balanced, ExtraTrees }
impl From<RandomForestModeArg> for RandomForestMode {
    fn from(value: RandomForestModeArg) -> Self {
        match value {
            RandomForestModeArg::Standard => RandomForestMode::Standard,
            RandomForestModeArg::Balanced => RandomForestMode::Balanced,
            RandomForestModeArg::ExtraTrees => RandomForestMode::ExtraTrees,
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum RandomForestMaxFeaturesArg { Sqrt, Log2, All, Half }
impl From<RandomForestMaxFeaturesArg> for RandomForestMaxFeatures {
    fn from(value: RandomForestMaxFeaturesArg) -> Self {
        match value {
            RandomForestMaxFeaturesArg::Sqrt => RandomForestMaxFeatures::Sqrt,
            RandomForestMaxFeaturesArg::Log2 => RandomForestMaxFeatures::Log2,
            RandomForestMaxFeaturesArg::All => RandomForestMaxFeatures::All,
            RandomForestMaxFeaturesArg::Half => RandomForestMaxFeatures::Half,
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum SvmKernelArg { Linear, Rbf, Polynomial, Sigmoid }
impl From<SvmKernelArg> for SvmKernel {
    fn from(value: SvmKernelArg) -> Self {
        match value {
            SvmKernelArg::Linear => SvmKernel::Linear,
            SvmKernelArg::Rbf => SvmKernel::Rbf,
            SvmKernelArg::Polynomial => SvmKernel::Polynomial,
            SvmKernelArg::Sigmoid => SvmKernel::Sigmoid,
        }
    }
}

#[derive(Debug)]
struct CliError(String);
impl Display for CliError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for CliError {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if let Some(path) = &cli.validate_yaml {
        let rules = RulesFile::from_yaml_path(path)?;
        println!("YAML is valid: {}", path.display());
        if cli.print_effective_yaml {
            print_rules_yaml(&rules)?;
        }
        return Ok(());
    }

    if cli.rules_yaml.is_none() && cli.classify_objects.is_none() && cli.method.is_none() {
        let mut cmd = Cli::command();
        cmd.print_help()?;
        println!();
        println!("\n{}", EXAMPLES);
        return Ok(());
    }

    let mut rules = if let Some(path) = &cli.rules_yaml {
        RulesFile::from_yaml_path(path)?
    } else {
        build_rules_from_cli(&cli)?
    };

    apply_overrides(&mut rules, &cli);
    rules.validate()?;

    if cli.print_effective_yaml {
        print_rules_yaml(&rules)?;
    }

    let classify_path = cli.classify_objects.clone().unwrap_or_else(|| PathBuf::from("."));
    let report = run_from_rules(&rules, &classify_path)?;

    if let Some(path) = &rules.csv_output {
        report.write_csv(path)?;
    }
    if let Some(path) = &rules.json_output {
        report.write_json(path)?;
    }

    print_report_summary(&report);
    Ok(())
}

fn build_rules_from_cli(cli: &Cli) -> Result<RulesFile, Box<dyn std::error::Error>> {
    let method = cli.method.clone().ok_or_else(|| CliError("--method is required when --rules-yaml is not used".into()))?;
    let nlp = cli.nlp.clone().ok_or_else(|| CliError("--nlp is required when --rules-yaml is not used".into()))?;
    let hot_path = cli.hot_path.clone().ok_or_else(|| CliError("--hot-path is required when --rules-yaml is not used".into()))?;
    let cold_path = cli.cold_path.clone().ok_or_else(|| CliError("--cold-path is required when --rules-yaml is not used".into()))?;

    let hot_label = cli.hot_label.clone().map(Into::into).unwrap_or(ClassificationLabel::WebAttack);
    let cold_label = cli.cold_label.clone().map(Into::into).unwrap_or(ClassificationLabel::RawData);

    let mut extra_match = Vec::new();
    if !cli.extra_match_path.is_empty() {
        let engine = cli.extra_match_engine.clone().unwrap_or(ExtraEngineArg::Regex);
        let score_add_points = cli.extra_match_score_add_points.unwrap_or(50.0);
        for path in &cli.extra_match_path {
            extra_match.push(ExtraMatchConfig {
                recursive_way: cli.recursive.clone().unwrap_or(OnOffArg::On).into(),
                engine: engine.clone().into(),
                path: path.clone(),
                score_add_points,
                title: None,
                description: None,
            });
        }
    }

    let rules = RulesFile {
        report_name: cli.report_name.clone(),
        method: method.into(),
        nlp: nlp.into(),
        threads: cli.threads,
        csv_output: cli.output_csv.clone(),
        json_output: cli.output_json.clone(),
        recursive_way: cli.recursive.clone().unwrap_or(OnOffArg::On).into(),
        hot_test_path: hot_path,
        cold_test_path: cold_path,
        hot_label: Some(hot_label),
        cold_label: Some(cold_label),
        score_sum: cli.score_sum.clone().unwrap_or(OnOffArg::Off).into(),
        extra_match,
        k: cli.k,
        p: cli.p,
        logistic_learning_rate: cli.logistic_learning_rate,
        logistic_epochs: cli.logistic_epochs,
        logistic_lambda: cli.logistic_lambda,
        random_forest_n_trees: cli.random_forest_n_trees,
        random_forest_mode: cli.random_forest_mode.clone().map(Into::into),
        random_forest_max_depth: cli.random_forest_max_depth,
        random_forest_max_features: cli.random_forest_max_features.clone().map(Into::into),
        random_forest_min_samples_split: cli.random_forest_min_samples_split,
        random_forest_min_samples_leaf: cli.random_forest_min_samples_leaf,
        random_forest_bootstrap: cli.random_forest_bootstrap,
        random_forest_oob_score: cli.random_forest_oob_score,
        svm_kernel: cli.svm_kernel.clone().map(Into::into),
        svm_c: cli.svm_c,
        svm_learning_rate: cli.svm_learning_rate,
        svm_epochs: cli.svm_epochs,
        svm_gamma: cli.svm_gamma,
        svm_degree: cli.svm_degree,
        svm_coef0: cli.svm_coef0,
        gradient_boosting_n_estimators: cli.gradient_boosting_n_estimators,
        gradient_boosting_learning_rate: cli.gradient_boosting_learning_rate,
        gradient_boosting_max_depth: cli.gradient_boosting_max_depth,
        isolation_forest_n_trees: cli.isolation_forest_n_trees,
        isolation_forest_contamination: cli.isolation_forest_contamination,
        isolation_forest_subsample_size: cli.isolation_forest_subsample_size,
    };
    Ok(rules)
}

fn apply_overrides(rules: &mut RulesFile, cli: &Cli) {
    if let Some(value) = &cli.report_name { rules.report_name = Some(value.clone()); }
    if let Some(value) = &cli.output_csv { rules.csv_output = Some(value.clone()); }
    if let Some(value) = &cli.output_json { rules.json_output = Some(value.clone()); }
    if let Some(value) = &cli.method { rules.method = value.clone().into(); }
    if let Some(value) = &cli.nlp { rules.nlp = value.clone().into(); }
    if let Some(value) = cli.threads { rules.threads = Some(value); }
    if let Some(value) = &cli.hot_path { rules.hot_test_path = value.clone(); }
    if let Some(value) = &cli.cold_path { rules.cold_test_path = value.clone(); }
    if let Some(value) = &cli.hot_label { rules.hot_label = Some(value.clone().into()); }
    if let Some(value) = &cli.cold_label { rules.cold_label = Some(value.clone().into()); }
    if let Some(value) = &cli.recursive { rules.recursive_way = value.clone().into(); }
    if let Some(value) = &cli.score_sum { rules.score_sum = value.clone().into(); }
    if cli.k.is_some() { rules.k = cli.k; }
    if cli.p.is_some() { rules.p = cli.p; }
    if cli.logistic_learning_rate.is_some() { rules.logistic_learning_rate = cli.logistic_learning_rate; }
    if cli.logistic_epochs.is_some() { rules.logistic_epochs = cli.logistic_epochs; }
    if cli.logistic_lambda.is_some() { rules.logistic_lambda = cli.logistic_lambda; }
    if cli.random_forest_n_trees.is_some() { rules.random_forest_n_trees = cli.random_forest_n_trees; }
    if cli.random_forest_mode.is_some() { rules.random_forest_mode = cli.random_forest_mode.clone().map(Into::into); }
    if cli.random_forest_max_depth.is_some() { rules.random_forest_max_depth = cli.random_forest_max_depth; }
    if cli.random_forest_max_features.is_some() { rules.random_forest_max_features = cli.random_forest_max_features.clone().map(Into::into); }
    if cli.random_forest_min_samples_split.is_some() { rules.random_forest_min_samples_split = cli.random_forest_min_samples_split; }
    if cli.random_forest_min_samples_leaf.is_some() { rules.random_forest_min_samples_leaf = cli.random_forest_min_samples_leaf; }
    if cli.random_forest_bootstrap.is_some() { rules.random_forest_bootstrap = cli.random_forest_bootstrap; }
    if cli.random_forest_oob_score.is_some() { rules.random_forest_oob_score = cli.random_forest_oob_score; }
    if cli.svm_kernel.is_some() { rules.svm_kernel = cli.svm_kernel.clone().map(Into::into); }
    if cli.svm_c.is_some() { rules.svm_c = cli.svm_c; }
    if cli.svm_learning_rate.is_some() { rules.svm_learning_rate = cli.svm_learning_rate; }
    if cli.svm_epochs.is_some() { rules.svm_epochs = cli.svm_epochs; }
    if cli.svm_gamma.is_some() { rules.svm_gamma = cli.svm_gamma; }
    if cli.svm_degree.is_some() { rules.svm_degree = cli.svm_degree; }
    if cli.svm_coef0.is_some() { rules.svm_coef0 = cli.svm_coef0; }
    if cli.gradient_boosting_n_estimators.is_some() { rules.gradient_boosting_n_estimators = cli.gradient_boosting_n_estimators; }
    if cli.gradient_boosting_learning_rate.is_some() { rules.gradient_boosting_learning_rate = cli.gradient_boosting_learning_rate; }
    if cli.gradient_boosting_max_depth.is_some() { rules.gradient_boosting_max_depth = cli.gradient_boosting_max_depth; }
    if cli.isolation_forest_n_trees.is_some() { rules.isolation_forest_n_trees = cli.isolation_forest_n_trees; }
    if cli.isolation_forest_contamination.is_some() { rules.isolation_forest_contamination = cli.isolation_forest_contamination; }
    if cli.isolation_forest_subsample_size.is_some() { rules.isolation_forest_subsample_size = cli.isolation_forest_subsample_size; }

    if !cli.extra_match_path.is_empty() {
        let engine = cli.extra_match_engine.clone().unwrap_or(ExtraEngineArg::Regex);
        let score_add_points = cli.extra_match_score_add_points.unwrap_or(50.0);
        rules.extra_match.clear();
        for path in &cli.extra_match_path {
            rules.extra_match.push(ExtraMatchConfig {
                recursive_way: rules.recursive_way,
                engine: engine.clone().into(),
                path: path.clone(),
                score_add_points,
                title: None,
                description: None,
            });
        }
    }
}

fn run_from_rules(rules: &RulesFile, classify_objects: &Path) -> Result<ClassificationReport, Box<dyn std::error::Error>> {
    rules.validate()?;
    if !classify_objects.exists() {
        return Err(Box::new(CliError(format!("classify_objects path does not exist: {}", classify_objects.display()))));
    }

    let classifier = ClassifierFactory::builder()
        .from_rules_file(rules)
        .build()?;

    let matchers = ScoringEngine::matchers_from_rules_file(rules)?;
    let mut report = ClassificationReport::new(
        rules.report_name.clone().unwrap_or_else(|| "Vec-Eyes CLI Report".to_string()),
    );

    let files = collect_files_recursively(classify_objects, rules.recursive_way.is_on())?;
    for file in files {
        let text = read_text_file(&file)?;
        let result = classifier.classify_text(&text, rules.score_sum, &matchers);

        let mut labels = Vec::new();
        for (label, score) in &result.labels {
            labels.push(format!("{}:{:.2}", label, score));
        }

        let mut hit_titles = Vec::new();
        for hit in &result.extra_hits {
            hit_titles.push(hit.title.clone());
        }

        let title_object = file.file_name().and_then(|x| x.to_str()).unwrap_or("object").to_string();
        let name_file_dataset = file.to_string_lossy().to_string();
        let score_percent = result.labels.first().map(|(_, score)| *score).unwrap_or(0.0);

        report.records.push(vec_eyes_lib::ClassificationRecord {
            title_object,
            name_file_dataset,
            classify_names_list: labels.join(","),
            date_of_occurrence: chrono::Utc::now(),
            score_percent,
            match_titles: hit_titles.join(","),
        });
    }

    Ok(report)
}

fn print_report_summary(report: &ClassificationReport) {
    println!("report_name={}", report.report_name);
    println!("records={}", report.records.len());
    println!();

    for record in &report.records {
        println!("object={}", record.title_object);
        println!("path={}", record.name_file_dataset);
        println!("labels={}", record.classify_names_list);
        println!("score_percent={:.2}", record.score_percent);
        println!("match_titles={}", record.match_titles);
        println!();
    }
}

fn print_rules_yaml(rules: &RulesFile) -> Result<(), Box<dyn std::error::Error>> {
    let rendered = serde_yaml::to_string(rules)?;
    println!("{}", rendered);
    Ok(())
}
