import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import IMDBDataset, prepare_data
from pathlib import Path
import time
import gc
from sklearn.metrics import classification_report


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model with multiple metrics

    Returns:
        dict: Dictionary containing various metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)

    # Get classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=["Negative", "Positive"],
        output_dict=True,
    )

    # Calculate additional metrics
    metrics = {
        "loss": avg_loss,
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "class_metrics": {
            "negative": {
                "precision": report["Negative"]["precision"],
                "recall": report["Negative"]["recall"],
                "f1": report["Negative"]["f1-score"],
            },
            "positive": {
                "precision": report["Positive"]["precision"],
                "recall": report["Positive"]["recall"],
                "f1": report["Positive"]["f1-score"],
            },
        },
    }

    return metrics


def print_metrics(metrics, split="Validation"):
    """Pretty print the metrics"""
    print(f"\n{split} Metrics:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision']:.4f}")
    print(f"Recall (macro): {metrics['recall']:.4f}")
    print(f"F1 Score (macro): {metrics['f1']:.4f}")

    print("\nPer-Class Metrics:")
    for class_name, class_metrics in metrics["class_metrics"].items():
        print(f"\n{class_name.capitalize()}:")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall: {class_metrics['recall']:.4f}")
        print(f"  F1 Score: {class_metrics['f1']:.4f}")


def train_model(
    model_name="distilbert-base-uncased",
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    max_length=256,
    num_samples=10000,
    output_dir="../../app/ml/models",
    freeze_base_model=True,
):
    """Train the sentiment analysis model

    Args:
        model_name: Name of the pretrained model to use
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        max_length: Maximum sequence length for tokenizer
        num_samples: Number of samples to use for training (10000 for quick experiments, 25000 for full training)
        output_dir: Directory to save the trained model
        freeze_base_model: Whether to freeze the base model layers (True for fine-tuning)
    """
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # Set memory allocation settings
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cudnn.benchmark = True

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # Freeze base model layers if specified ('True' for fine-tuning!)
    if freeze_base_model:
        print("\nFreezing base model layers...")
        # Freeze the base model layers
        for param in model.distilbert.parameters():
            param.requires_grad = False

        # Keep classification layers trainable
        for param in model.pre_classifier.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Print parameter statistics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        print("- Base DistilBERT layers: Frozen")
        print("- Pre-classifier layer: Trainable")
        print("- Classifier layer: Trainable")
    else:
        print("\nTraining all layers...")
        trainable_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)

    # Prepare data
    train_data, val_data = prepare_data(num_samples=num_samples)

    print("\nCreating datasets...")
    train_dataset = IMDBDataset(
        train_data["text"], train_data["label"], tokenizer, max_length
    )
    val_dataset = IMDBDataset(
        val_data["text"], val_data["label"], tokenizer, max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )

    total_steps = (
        len(train_loader) // 4
    ) * num_epochs  # Using gradient accumulation steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_accuracy = 0
    start_time = time.time()

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Training on device: {device}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()

        # Training
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / 4  # gradient accumulation steps
            loss.backward()

            if (i + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * 4

            # Clear cache periodically
            if i % 100 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation
        metrics = evaluate_model(model, val_loader, device)
        print_metrics(metrics)

        # Save best model
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            print(f"New best accuracy: {best_accuracy:.4f}! Saving model...")
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")

        epoch_time = time.time() - epoch_start
        print(f"Epoch time: {epoch_time:.2f} seconds")

        # Clear cache between epochs
        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Best accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    train_model()
