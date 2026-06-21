"""Choice model training script — trains two-tower model with BCE loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from applications.choice.model import ChoiceModel


def get_batches(df, batch_size: int = 256):
    """Yield batches of data from dataframe."""
    n_samples = len(df)
    for i in range(0, n_samples, batch_size):
        yield df.iloc[i : i + batch_size]


def train(
    data_path: Path = Path("applications/choice/choice_training.parquet"),
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    device: str = "cpu",
    save_path: Path = Path("applications/choice/choice_head.pt"),
) -> None:
    """Train choice model with participant-level 70/30 split."""
    print(f"Loading training data from {data_path}")
    table = pq.read_table(data_path)
    df = table.to_pandas()
    print(f"Loaded {len(df)} choice rows")

    # Extract features
    participants = df["participant_id"].unique()

    # Participant-level split
    train_pids, val_pids = train_test_split(
        participants, test_size=0.3, random_state=42
    )

    train_df = df[df["participant_id"].isin(train_pids)]
    val_df = df[df["participant_id"].isin(val_pids)]

    print(f"Train: {len(train_df)} rows ({len(train_pids)} participants)")
    print(f"Val:   {len(val_df)} rows ({len(val_pids)} participants)")

    # Determine product_dim from data
    sample_features = train_df["product_features"].iloc[0]
    product_dim = len(sample_features)
    print(f"Product feature dimension: {product_dim}")

    # Build model
    model = ChoiceModel(
        cdt_dim=128,
        product_dim=product_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    max_patience = 10

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        try:
            for batch_data in get_batches(train_df, batch_size):
                # Extract features
                cdt_embeddings = np.stack(batch_data["cdt_embedding"].tolist())
                product_features = np.stack(batch_data["product_features"].tolist())
                labels = torch.tensor(batch_data["chosen"].tolist()).float()

                # Convert to tensors
                cdt_tensor = torch.tensor(cdt_embeddings).float()
                prod_tensor = torch.tensor(product_features).float()

                probs = model(cdt_tensor, prod_tensor).squeeze(1)
                loss = criterion(probs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
        except Exception as e:
            print(f"Error at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            raise

        train_loss /= len(train_df) / batch_size

        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = []
            val_labels = []

            for batch_data in get_batches(val_df, batch_size):
                cdt_embeddings = np.stack(batch_data["cdt_embedding"].tolist())
                product_features = np.stack(batch_data["product_features"].tolist())
                labels = batch_data["chosen"].tolist()

                cdt_tensor = torch.tensor(cdt_embeddings).float()
                prod_tensor = torch.tensor(product_features).float()

                probs = model(cdt_tensor, prod_tensor).squeeze(1)
                val_probs.extend(probs.tolist())
                val_labels.extend(labels)

        val_auc = roc_auc_score(val_labels, val_probs)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_auc={val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=True)
            print(f"  ✅ Best model saved (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print("Training complete")
    print(f"Best val AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M1 choice model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("applications/choice/choice_training.parquet"),
        help="Path to choice training data",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--device", default="cpu", help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("applications/choice/choice_head.pt"),
        help="Path to save model checkpoint",
    )

    args = parser.parse_args()

    try:
        train(
            data_path=args.data,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            device=args.device,
            save_path=args.save,
        )
    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)


if __name__ == "__main__":
    main()
