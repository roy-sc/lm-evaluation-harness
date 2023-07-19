import pandas as pd
from datasets import DatasetDict, Dataset


def get_dataset_dict_from_jsonl(json_file_path: str) -> DatasetDict:
    df = pd.read_json(json_file_path, lines=True)
    df_test = Dataset.from_pandas(df[df.split == 'test'])
    df_val = Dataset.from_pandas(df[df.split == 'valid'])
    dataset_dict = DatasetDict({
         'validation': df_val,
         'test': df_test
     })
    return dataset_dict


def main():
    dataset_path = 'lm_eval/datasets/frank/human_annotations_sentence_factual.json'
    dataset_dict = get_dataset_dict_from_jsonl(dataset_path)
    hub_save_name = f"mtc/frank-test-set-with-factuality-annotation"
    dataset_dict.push_to_hub(hub_save_name)


if __name__ == '__main__':
    main()
