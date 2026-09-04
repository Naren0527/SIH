"""
SignSync GPU-optimized BiLSTM trainer.

Use this after running the body-relative preprocessing script.

Key points:
- Uses RTX 4050 through CUDA.
- Mixed precision on CUDA.
- Pinned-memory DataLoaders.
- Multiple CPU DataLoader workers.
- Larger batch size suitable for this small BiLSTM.
- Keeps the exact 258-feature / 60-frame architecture.
- Mean-pools all LSTM timesteps, matching realtime inference.
- Augmentation is designed for BODY-RELATIVE coordinates.
"""

import os
import json
import random
import time

# Keep CPU-side numerical libraries from oversubscribing the machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed_sequences"
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "signsync_bilstm.pt"
)

NORMALIZATION_PATH = os.path.join(
    MODEL_DIR,
    "normalization.npz"
)

CLASS_MAPPING_PATH = os.path.join(
    DATA_DIR,
    "class_mapping.json"
)

TRAINING_INFO_PATH = os.path.join(
    MODEL_DIR,
    "training_info.json"
)

SEQUENCE_LENGTH = 60
INPUT_SIZE = 258

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.30

# RTX 4050 6 GB. 128 is usually safe for this tiny model.
BATCH_SIZE = 128

EPOCHS = 50
LEARNING_RATE = 0.0007
WEIGHT_DECAY = 5e-4
RANDOM_SEED = 42

# Windows CPU workers. 4 is a safe starting point.
# Override:
#   $env:SIGNSYNC_LOADER_WORKERS=6
LOADER_WORKERS = int(
    os.environ.get(
        "SIGNSYNC_LOADER_WORKERS",
        "4"
    )
)

AUGMENTATION_PROBABILITY = 0.90

# Body-relative coordinates are centered around the shoulder origin.
# DO NOT use the old 0.5-centered image-coordinate augmentation.
BODY_SCALE_RANGE = 0.10

# Small detector noise.
LANDMARK_NOISE_STD = 0.008

# Occasional missing landmark simulation.
FEATURE_DROPOUT_PROBABILITY = 0.012

# Temporal variation.
TEMPORAL_JITTER_STD = 1.25


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# HELPERS
# ============================================================

def save_json(path, data):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=2
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# BODY-RELATIVE AUGMENTATION
# ============================================================

def augment_sequence(sequence):
    """
    Augmentation for body-relative coordinates.

    Coordinates are centered around the shoulder origin, so translation
    around 0.5 is WRONG here.

    We instead apply:
      1. small global scale around the origin
      2. small landmark noise
      3. temporal jitter
      4. feature dropout

    Left/right semantic slots are never swapped.
    """

    sequence = sequence.copy().astype(
        np.float32
    )

    if random.random() > AUGMENTATION_PROBABILITY:
        return sequence

    # --------------------------------------------------------
    # 1. Small global scale around body origin.
    # --------------------------------------------------------

    scale = random.uniform(
        1.0 - BODY_SCALE_RANGE,
        1.0 + BODY_SCALE_RANGE
    )

    # x, y, z coordinates only.
    for start in (0, 63):
        for col in range(
            start,
            start + 63,
            3
        ):
            sequence[:, col:col + 3] *= scale

    for col in range(
        126,
        258,
        4
    ):
        sequence[:, col:col + 3] *= scale

    # --------------------------------------------------------
    # 2. Small landmark noise.
    # --------------------------------------------------------

    noise = np.random.normal(
        0.0,
        LANDMARK_NOISE_STD,
        size=sequence.shape
    ).astype(
        np.float32
    )

    # Never alter visibility.
    noise[:, 129::4] = 0.0

    sequence += noise

    # --------------------------------------------------------
    # 3. Temporal jitter.
    # --------------------------------------------------------

    base_indices = np.arange(
        SEQUENCE_LENGTH,
        dtype=np.float32
    )

    jitter = np.random.normal(
        0.0,
        TEMPORAL_JITTER_STD,
        size=SEQUENCE_LENGTH
    ).astype(
        np.float32
    )

    jitter[0] = 0.0
    jitter[-1] = 0.0

    indices = np.rint(
        base_indices + jitter
    ).astype(
        np.int64
    )

    indices = np.clip(
        indices,
        0,
        SEQUENCE_LENGTH - 1
    )

    indices = np.maximum.accumulate(
        indices
    )

    indices[-1] = SEQUENCE_LENGTH - 1

    sequence = sequence[indices]

    # --------------------------------------------------------
    # 4. Small feature dropout.
    # --------------------------------------------------------

    mask = (
        np.random.random(sequence.shape)
        < FEATURE_DROPOUT_PROBABILITY
    )

    # Do not randomly destroy visibility.
    mask[:, 129::4] = False

    sequence[mask] = 0.0

    return sequence.astype(
        np.float32
    )


# ============================================================
# DATASET
# ============================================================

class SignDataset(Dataset):

    def __init__(
        self,
        samples,
        mean,
        std,
        augment=False
    ):
        self.samples = samples
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        sequence = np.load(
            path
        ).astype(
            np.float32
        )

        if self.augment:
            sequence = augment_sequence(
                sequence
            )

        sequence = (
            sequence - self.mean
        ) / self.std

        sequence = np.nan_to_num(
            sequence,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return (
            torch.from_numpy(sequence),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# MODEL
# ============================================================

class SignBiLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_size,
                256
            ),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(
                hidden_size * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.input_projection(x)

        output, _ = self.lstm(x)

        # IMPORTANT:
        # Match realtime inference: mean across all 60 timesteps.
        x = output.mean(dim=1)

        return self.classifier(x)


# ============================================================
# CLASS MAPPING
# ============================================================

def load_class_mapping():

    if not os.path.isfile(
        CLASS_MAPPING_PATH
    ):
        raise FileNotFoundError(
            "class_mapping.json not found:\n"
            f"{CLASS_MAPPING_PATH}"
        )

    with open(
        CLASS_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# ============================================================
# LOAD SAMPLES
# ============================================================

def load_samples(class_mapping):

    samples = []

    for class_name, label in sorted(
        class_mapping.items(),
        key=lambda item: item[1]
    ):

        class_dir = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.isdir(
            class_dir
        ):
            print(
                f"WARNING: Missing class directory: "
                f"{class_name}"
            )
            continue

        files = [
            os.path.join(
                class_dir,
                filename
            )
            for filename in os.listdir(
                class_dir
            )
            if filename.lower().endswith(
                ".npy"
            )
        ]

        files.sort()

        for path in files:
            samples.append(
                (
                    path,
                    int(label)
                )
            )

    return samples


# ============================================================
# VALIDATE
# ============================================================

def validate_samples(samples):

    valid = []

    print(
        "Checking processed sequences..."
    )

    for path, label in samples:

        try:
            data = np.load(
                path,
                mmap_mode="r"
            )

            if data.shape != (
                SEQUENCE_LENGTH,
                INPUT_SIZE
            ):
                continue

            if not np.isfinite(
                data
            ).all():
                continue

            valid.append(
                (
                    path,
                    label
                )
            )

        except Exception:
            continue

    print(
        f"Valid sequences: {len(valid)}"
    )

    return valid


# ============================================================
# SPLIT
# ============================================================

def create_splits(samples):

    labels = np.array(
        [
            label
            for _, label in samples
        ]
    )

    indices = np.arange(
        len(samples)
    )

    train_indices, temp_indices = (
        train_test_split(
            indices,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=labels
        )
    )

    temp_labels = labels[
        temp_indices
    ]

    val_indices, test_indices = (
        train_test_split(
            temp_indices,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=temp_labels
        )
    )

    return (
        [samples[i] for i in train_indices],
        [samples[i] for i in val_indices],
        [samples[i] for i in test_indices]
    )


# ============================================================
# NORMALIZATION
# ============================================================

def calculate_normalization(samples):

    print(
        "Calculating training normalization..."
    )

    total_sum = np.zeros(
        INPUT_SIZE,
        dtype=np.float64
    )

    total_squared = np.zeros(
        INPUT_SIZE,
        dtype=np.float64
    )

    total_count = 0

    for index, (path, _) in enumerate(
        samples,
        start=1
    ):

        data = np.load(
            path
        ).astype(
            np.float64
        )

        total_sum += data.sum(
            axis=0
        )

        total_squared += np.square(
            data
        ).sum(
            axis=0
        )

        total_count += data.shape[0]

        if index % 500 == 0:
            print(
                f"  Normalization: "
                f"{index}/{len(samples)}"
            )

    mean = (
        total_sum
        / total_count
    )

    variance = (
        total_squared
        / total_count
    ) - np.square(mean)

    variance = np.maximum(
        variance,
        1e-8
    )

    std = np.sqrt(
        variance
    )

    std = np.maximum(
        std,
        1e-6
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32)
    )


# ============================================================
# ACCURACY
# ============================================================

@torch.no_grad()
def calculate_accuracy(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0

    for sequences, labels in loader:

        sequences = sequences.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        if USE_AMP:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                outputs = model(
                    sequences
                )
        else:
            outputs = model(
                sequences
            )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        correct / total
        if total
        else 0.0
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:

        sequences = sequences.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                outputs = model(
                    sequences
                )

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            outputs = model(
                sequences
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        total_loss += (
            loss.detach().item()
            * labels.size(0)
        )

        correct += (
            outputs.argmax(dim=1)
            == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    class_mapping,
    mean,
    std,
    epoch,
    val_accuracy
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "input_size":
            INPUT_SIZE,

        "sequence_length":
            SEQUENCE_LENGTH,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "dropout":
            DROPOUT,

        "num_classes":
            len(class_mapping),

        "class_mapping":
            class_mapping,

        "epoch":
            epoch,

        "validation_accuracy":
            val_accuracy,

        "representation":
            "body_relative",

        "pooling":
            "mean",

        "device":
            str(DEVICE)
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    np.savez(
        NORMALIZATION_PATH,
        mean=mean,
        std=std
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print("=" * 72)
    print("SignSync GPU BiLSTM Training")
    print("=" * 72)

    print(
        f"Device          : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

        props = torch.cuda.get_device_properties(0)

        print(
            f"VRAM            : "
            f"{props.total_memory / 1024**3:.2f} GB"
        )

        print(
            f"CUDA            : "
            f"{torch.version.cuda}"
        )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"DataLoader CPUs : {LOADER_WORKERS}"
    )

    print(
        f"AMP             : {USE_AMP}"
    )

    print(
        "Representation  : BODY-RELATIVE"
    )

    print()

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "Do not train this GPU version until "
            "torch.cuda.is_available() returns True."
        )

    # --------------------------------------------------------
    # Mapping
    # --------------------------------------------------------

    class_mapping = load_class_mapping()

    num_classes = len(
        class_mapping
    )

    print(
        f"Classes: {num_classes}"
    )

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    samples = load_samples(
        class_mapping
    )

    print(
        f"Sequences found: {len(samples)}"
    )

    samples = validate_samples(
        samples
    )

    if len(samples) < num_classes * 2:
        raise RuntimeError(
            "Not enough valid training data."
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_samples,
        val_samples,
        test_samples
    ) = create_splits(
        samples
    )

    print()
    print("Dataset split:")
    print(
        f"Train:      {len(train_samples)}"
    )
    print(
        f"Validation: {len(val_samples)}"
    )
    print(
        f"Test:       {len(test_samples)}"
    )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    mean, std = calculate_normalization(
        train_samples
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    np.savez(
        NORMALIZATION_PATH,
        mean=mean,
        std=std
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = SignDataset(
        train_samples,
        mean,
        std,
        augment=True
    )

    val_dataset = SignDataset(
        val_samples,
        mean,
        std,
        augment=False
    )

    test_dataset = SignDataset(
        test_samples,
        mean,
        std,
        augment=False
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    common_loader_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": LOADER_WORKERS,
        "pin_memory": True,
        "persistent_workers": (
            LOADER_WORKERS > 0
        )
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **common_loader_args
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_args
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_args
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SignBiLSTM(
        INPUT_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS,
        num_classes,
        DROPOUT
    ).to(
        DEVICE
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = 0.0
    best_epoch = 0
    history = []

    print()
    print("=" * 72)
    print("TRAINING ON RTX 4050")
    print("=" * 72)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.perf_counter()

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler
            )
        )

        val_accuracy = calculate_accuracy(
            model,
            val_loader
        )

        scheduler.step(
            val_accuracy
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            peak = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            torch.cuda.reset_peak_memory_stats()

            gpu_text = (
                f" | VRAM {allocated:.2f}/"
                f"{peak:.2f} GB"
            )

        else:
            gpu_text = ""

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {train_loss:.4f} | "
            f"Train {train_accuracy * 100:.2f}% | "
            f"Val {val_accuracy * 100:.2f}% | "
            f"LR {current_lr:.6f} | "
            f"{epoch_time:.1f}s"
            f"{gpu_text}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy,
                "learning_rate": current_lr,
                "epoch_seconds": epoch_time
            }
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            save_checkpoint(
                model,
                class_mapping,
                mean,
                std,
                epoch,
                val_accuracy
            )

            print(
                "  Best model saved."
            )

        if (
            epoch - best_epoch
            >= 10
        ):

            print(
                "Early stopping."
            )

            break

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    final_val_accuracy = calculate_accuracy(
        model,
        val_loader
    )

    test_accuracy = calculate_accuracy(
        model,
        test_loader
    )

    # --------------------------------------------------------
    # Save info
    # --------------------------------------------------------

    training_info = {
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "num_classes": num_classes,
        "total_sequences": len(samples),
        "train_sequences": len(train_samples),
        "validation_sequences": len(val_samples),
        "test_sequences": len(test_samples),
        "sequence_length": SEQUENCE_LENGTH,
        "input_features": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "loader_workers": LOADER_WORKERS,
        "mixed_precision": USE_AMP,
        "epochs_configured": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": 0.05,
        "representation": "body_relative",
        "pooling": "mean",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "final_validation_accuracy": final_val_accuracy,
        "test_accuracy": test_accuracy,
        "history": history
    }

    save_json(
        TRAINING_INFO_PATH,
        training_info
    )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"Best validation accuracy : "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Final validation accuracy: "
        f"{final_val_accuracy * 100:.2f}%"
    )

    print(
        f"Test accuracy            : "
        f"{test_accuracy * 100:.2f}%"
    )

    print()
    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Normalization: "
        f"{NORMALIZATION_PATH}"
    )

    print(
        f"Training info: "
        f"{TRAINING_INFO_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
