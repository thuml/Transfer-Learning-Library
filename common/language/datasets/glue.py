from torch.utils.data import Dataset
from datasets import load_dataset
from transformers.utils.versions import require_version

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


class GLUE(Dataset):
    task_to_keys = {
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
        "snli": ("premise", "hypothesis"),
    }
    CLASSES = ["entailment", "not_entailment"]

    def __init__(self, task: str, transform, split='train'):
        super(GLUE, self).__init__()
        assert task in self.task_to_keys.keys()
        self.task = task
        if task == "snli":
            raw_datasets = load_dataset("snli")
        else:
            raw_datasets = load_dataset("glue", task)
        self.classes = self.CLASSES
        sentence1_key, sentence2_key = self.task_to_keys[task]

        raw_datasets = raw_datasets.filter(lambda x: x["label"] != -1)

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
            )
            result = transform(*texts)

            labels = examples["label"]
            labels = [1 if l == 2 else l for l in labels]
            if task == "qqp":
                labels = [1-l for l in labels]
            examples["label"] = labels
            if "label" in examples:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

            return result

        self.datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )[split]

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self) -> int:
        return len(self.datasets)

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)
