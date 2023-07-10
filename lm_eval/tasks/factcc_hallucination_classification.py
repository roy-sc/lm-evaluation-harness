from functools import partial

from lm_eval.base import Task, rf
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score


def _complex_metric(preds, labels, average="micro", metric="bacc"):
    match metric:
        case "bacc":
            return balanced_accuracy_score(y_true=labels, y_pred=preds)
        case "f1":
            return f1_score(y_true=labels, y_pred=preds, average=average)
        case "precision":
            return precision_score(y_true=labels, y_pred=preds, average=average)
        case "recall":
            return recall_score(y_true=labels, y_pred=preds, average=average)
        case _:
            raise ValueError(f" Unknown metric {metric}")


def _complex_metric_agg(metric, items):
    predictions, references = zip(*items)

    return _complex_metric(preds=predictions, labels=references, average='binary', metric=metric)


class FactCCHallucinationClassificationTask(Task):
    VERSION = 0
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "mtc/factcc_annotated_eval_data"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return (f"""Given an article and a corresponding claim, label the claim as CORRECT, if the claim only contains information present in the article. Otherwise, label the claim as INCORRECT:
Article: {doc['text']}
Claim: {doc['claim']}
Label:
""")

    def doc_to_target(self, doc):
        return " " + doc['label']

    def construct_requests(self, doc, ctx):
        continuation = rf.greedy_until(ctx, {"until": ["\n"]})
        return continuation

    def process_results(self, doc, results):
        prediction = results
        truth = doc["label"]
        return {"bacc": (prediction, truth),
                "f1": (prediction, truth),
                "precision": (prediction, truth),
                "recall": (prediction, truth)}

    def aggregation(self):
        return {
            "bacc": partial(
                _complex_metric_agg, "bacc"
            ),
            "f1": partial(
                _complex_metric_agg, "f1"
            ),
            "precision": partial(
                _complex_metric_agg, "precision"
            ),
            "recall": partial(
                _complex_metric_agg, "recall"
            )
        }

    def higher_is_better(self):
        return {"bacc": True,
                "f1": True,
                "precision": True,
                "recall": True}
