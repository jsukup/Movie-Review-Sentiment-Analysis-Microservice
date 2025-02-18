from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from collections import Counter
import pandas as pd


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def print_label_distribution(data, split_name):
    """Print the distribution of labels in a dataset"""
    labels = data["label"]
    label_counts = Counter(labels)
    total = len(labels)
    print(f"\n{split_name} Label Distribution:")
    for label, count in sorted(label_counts.items()):
        label_name = "Positive" if label == 1 else "Negative"
        percentage = (count / total) * 100
        print(f"{label_name}: {count} ({percentage:.2f}%)")


def prepare_data(num_samples=None, test_size=0.2, random_state=42):
    """Load and prepare IMDB dataset

    Args:
        num_samples: If None, uses full dataset (25,000). Otherwise, uses specified number.
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
    """
    print(
        f"Loading IMDB dataset{f' (using {num_samples} samples)' if num_samples else ''}"
    )

    # Load the full dataset first
    dataset = load_dataset("imdb", split="train")

    # Convert to pandas for easier handling
    df = pd.DataFrame({"text": dataset["text"], "label": dataset["label"]})

    if num_samples:
        # Stratified sampling to maintain label distribution
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(n=num_samples // 2, random_state=random_state))
            .reset_index(drop=True)
        )

    # Split into train and validation while preserving label distribution
    train_data, val_data = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    # Convert back to dictionary format
    train_dict = {
        "text": train_data["text"].tolist(),
        "label": train_data["label"].tolist(),
    }
    val_dict = {"text": val_data["text"].tolist(), "label": val_data["label"].tolist()}

    # Print distributions
    print_label_distribution(train_dict, "Training")
    print_label_distribution(val_dict, "Validation")

    print(
        f"\nTrain size: {len(train_dict['text'])}, Validation size: {len(val_dict['text'])}"
    )
    return train_dict, val_dict
