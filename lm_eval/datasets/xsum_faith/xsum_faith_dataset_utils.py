import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset

from lm_eval.datasets.utils import print_extractiveness_data_for_jsonl


def load_xsum_dataset():
    test_dataset = load_dataset("xsum", split="test")
    test_dataset.to_json("lm_eval/datasets/xsum_faith/xsum_test.jsonl")


def get_dataset_dict_from_jsonl(json_file_path: str) -> DatasetDict:
    df = pd.read_json(json_file_path, lines=True)
    df_test = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({
        'test': df_test
    })
    return dataset_dict


def analyze_extractiveness():
    original_text_key = "article"
    summary_key = "summary"
    factuality_key = "is_factual"

    root_path = "lm_eval/datasets/xsum_faith/xsum_faith_dataset_processed.jsonl"
    show_only_positive = True
    print_extractiveness_data_for_jsonl(root_path, show_only_for_positive=show_only_positive,
                                        article_key=original_text_key, summary_key=summary_key,
                                        factuality_key=factuality_key)


def push_to_hub():
    dataset_path = 'lm_eval/datasets/xsum_faith/xsum_faith_dataset_processed.jsonl'
    dataset_dict = get_dataset_dict_from_jsonl(dataset_path)
    hub_save_name = f"mtc/xsum-faith-test-set-with-factuality-annotation"
    dataset_dict.push_to_hub(hub_save_name)


def main():
    analyze_extractiveness()


if __name__ == '__main__':
    main()
