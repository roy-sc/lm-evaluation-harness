import statistics
from typing import Dict

import pandas as pd
from tqdm import tqdm

from lm_eval.datasets.fragments import Fragments


def get_extractiveness_metrics(df: pd.DataFrame, language: str, article_key: str, summary_key: str) -> Dict:
    extractiveness_metrics_dict = {}
    coverage, density, compression = [], [], []

    for index, row in tqdm(df.iterrows()):
        original_sentence = row[article_key]
        paraphrase = row[summary_key]
        fragment = Fragments(original_sentence, paraphrase, language=language)
        coverage.append(fragment.coverage())
        density.append(fragment.density())
        compression.append(fragment.compression())

    extractiveness_metrics_dict["coverage"] = coverage
    extractiveness_metrics_dict["density"] = density
    extractiveness_metrics_dict["compression"] = compression
    return extractiveness_metrics_dict


def print_extractiveness_averages(extractiveness_metrics_dict: Dict):
    for metric, values in extractiveness_metrics_dict.items():
        print(f"Mean value for {metric}: {statistics.mean(values)} ")


def print_extractiveness_data_for_jsonl(jsonl_path: str, article_key: str, summary_key: str, factuality_key: str,
                                        show_only_for_positive: bool = False):
    print(f"Extractive Metrics for: {jsonl_path}")
    df = pd.read_json(jsonl_path, lines=True)
    if show_only_for_positive:
        df = df[df[factuality_key]]
    extractiveness_metrics_dict = get_extractiveness_metrics(df, language='english', article_key=article_key,
                                                             summary_key=summary_key)
    print_extractiveness_averages(extractiveness_metrics_dict)
