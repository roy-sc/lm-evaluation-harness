from functools import partial

import numpy as np

from lm_eval.base import Task, rf
from lm_eval.metrics import complex_metric_agg


class SeahorseClassificationTask(Task):
    VERSION = 0
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "mtc/seahorse_dataset_with_articles"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    default_prompt_template = """### System:
You are StableBeluga, an AI that follows instructions extremely well. Help as much as you can.

### User: Given an article and its summary in {lang}, determine if all information in the summary is sourced from the article. Return True if it is, and False if not. 
Article: {article}
Summary: {summary}

### Assistant:"
"""

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():

            if self._training_docs is None:

                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():

            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():

            return self.dataset["test"]

    def doc_to_text(self, doc):
        if not self.prompt_template:
            self.prompt_template = self.default_prompt_template
        prompt = self.prompt_template.format(article=doc['article'],
                                             summary=doc['summary'], label="", lang=doc['worker_lang'])
        return prompt

    def doc_to_target(self, doc):
        label = str(doc["question4"] == "Yes")
        return " " + label

    def construct_requests(self, doc, ctx):
        ll_false, _ = rf.loglikelihood(ctx, " False")
        ll_true, _ = rf.loglikelihood(ctx, " True")
        return ll_false, ll_true

    @staticmethod
    def convert_label(label):
        label = label.strip()
        if label.lower() == 'false':
            return 0
        elif label.lower() == 'true':
            return 1
        else:
            raise ValueError("Invalid label!")

    def process_results(self, doc, results):
        prediction = np.argmax(results)
        truth = self.convert_label(self.doc_to_target(doc))
        print(f"Results: {results}, Prediction {prediction}, Truth: {truth}")
        return {"bacc": (prediction, truth),
                "f1": (prediction, truth),
                "precision": (prediction, truth),
                "recall": (prediction, truth)}

    def aggregation(self):
        return {
            "bacc": partial(
                complex_metric_agg, "bacc"
            ),
            "f1": partial(
                complex_metric_agg, "f1"
            ),
            "precision": partial(
                complex_metric_agg, "precision"
            ),
            "recall": partial(
                complex_metric_agg, "recall"
            )
        }

    def higher_is_better(self):
        return {"bacc": True,
                "f1": True,
                "precision": True,
                "recall": True}
