
def convert_from_wilds_dataset(wild_dataset):
    class Dataset:
        def __init__(self):
            self.dataset = wild_dataset

        def __getitem__(self, idx):
            x, y, metadata = self.dataset[idx]
            x["labels"] = y
            return x

        def __len__(self):
            return len(self.dataset)

    return Dataset()


def get_dataset(dataset, source, target, transform, root_dir=None):
    train_source_dataset, train_target_dataset, val_dataset, num_classes = None, None, None, 1
    dataset = dataset.lower()
    if dataset == "glue":
        from common.language.datasets.glue import GLUE
        train_source_dataset = GLUE(task=source, transform=transform, split="train")
        train_target_dataset = GLUE(task=target, transform=transform, split="train")
        val_dataset = GLUE(task=target, transform=transform,
                              split="validation_matched" if target == "mnli" else "validation")
        num_classes = 2
    elif dataset == "amazon":
        from wilds.datasets.amazon_dataset import AmazonDataset
        # amazon = AmazonDataset(version=None, root_dir=root_dir)
        # train_source_dataset = convert_from_wilds_dataset(amazon.get_subset('train', transform=transform))
        # train_target_dataset = val_dataset = convert_from_wilds_dataset(amazon.get_subset('val', transform=transform))
        source_dataset = AmazonDataset(version=None, root_dir=root_dir, split_scheme="{}_generalization".format(source))
        train_source_dataset = convert_from_wilds_dataset(source_dataset.get_subset('train', transform=transform))
        target_dataset = AmazonDataset(version=None, root_dir=root_dir, split_scheme="{}_generalization".format(target))
        train_target_dataset = val_dataset = convert_from_wilds_dataset(target_dataset.get_subset('train', transform=transform))
        num_classes = 5
    elif dataset == "civilcomments":
        from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
        civilcomments = CivilCommentsDataset(version=None, root_dir=root_dir)
        train_source_dataset = convert_from_wilds_dataset(civilcomments.get_subset('train', transform=transform))
        train_target_dataset = val_dataset = convert_from_wilds_dataset(civilcomments.get_subset('val', transform=transform))
        num_classes = 2
    else:
        raise NotImplementedError(dataset)
    return train_source_dataset, train_target_dataset, val_dataset, num_classes

