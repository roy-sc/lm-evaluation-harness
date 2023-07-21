from lm_eval.datasets.utils import print_extractiveness_data_for_jsonl


def main():
    original_text_key = "article"
    summary_key = "summary"
    factuality_key = "Factual"

    root_path = "lm_eval/datasets/frank/human_annotations_sentence_factual.jsonl"
    show_only_positive = True
    print_extractiveness_data_for_jsonl(root_path, show_only_for_positive=show_only_positive,
                                        article_key=original_text_key, summary_key=summary_key,
                                        factuality_key=factuality_key)


if __name__ == '__main__':
    main()
