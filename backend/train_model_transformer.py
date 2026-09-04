
"""
SignSync Transformer trainer.

Uses the EXISTING 60x258 body-relative processed dataset.
Does NOT redo MediaPipe preprocessing.

Architecture:
    258 landmarks
        -> 256 projection
        -> learned positional embeddings
        -> 3-layer Transformer Encoder
        -> learned attention pooling
        -> 256 -> 128 -> 61 classes

Designed for an RTX 4050.
"""

import os
import json
import random
import time

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
    PROJECT_ROOT, "data", "processed_sequences"
)

DATA_DIR = os.path.join(
    PROJECT_ROOT, "data"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT, "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR, "signsync_transformer.pt"
)

NORMALIZATION_PATH = os.path.join(
    MODEL_DIR, "normalization.npz"
)

CLASS_MAPPING_PATH = os.path.join(
    DATA_DIR, "class_mapping.json"
)

TRAINING_INFO_PATH = os.path.join(
    MODEL_DIR, "transformer_training_info.json"
)

SEQUENCE_LENGTH = 60
INPUT_SIZE = 258
NUM_CLASSES = 61

D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.20

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05

RANDOM_SEED = 42

LOADER_WORKERS = int(
    os.environ.get(
        "SIGNSYNC_LOADER_WORKERS",
        "4"
    )
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, data):
    os.makedirs(
        os.path.dirname(path),
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


# ============================================================
# AUGMENTATION
# ============================================================

def temporal_warp(sequence):
    """
    Resample a 60-frame sign sequence with small random
    speed variation while keeping exactly 60 frames.
    """

    length = sequence.shape[0]

    if length < 3:
        return sequence

    speed = random.uniform(
        0.80,
        1.20
    )

    center = (length - 1) / 2.0

    source = (
        center
        + (np.arange(length) - center)
        * speed
    )

    source = np.clip(
        source,
        0,
        length - 1
    )

    left = np.floor(source).astype(
        np.int64
    )

    right = np.ceil(source).astype(
        np.int64
    )

    alpha = (
        source - left
    ).astype(
        np.float32
    )[:, None]

    warped = (
        sequence[left]
        * (1.0 - alpha)
        + sequence[right]
        * alpha
    )

    return warped.astype(
        np.float32
    )


def augment_sequence(sequence):

    sequence = sequence.copy().astype(
        np.float32
    )

    # Small body-relative scale.
    scale = random.uniform(
        0.90,
        1.10
    )

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

    # Landmark noise.
    noise = np.random.normal(
        0.0,
        0.006,
        size=sequence.shape
    ).astype(
        np.float32
    )

    # Never alter pose visibility.
    noise[:, 129::4] = 0.0

    sequence += noise

    # Temporal speed variation.
    if random.random() < 0.75:
        sequence = temporal_warp(
            sequence
        )

    # Random short frame masking.
    if random.random() < 0.30:

        width = random.randint(
            1,
            3
        )

        start = random.randint(
            0,
            SEQUENCE_LENGTH - width
        )

        sequence[
            start:start + width
        ] = 0.0

    # Small feature dropout.
    if random.random() < 0.50:

        mask = (
            np.random.random(
                sequence.shape
            ) < 0.008
        )

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
        ).astype(
            np.float32
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

class SignTransformer(nn.Module):

    def __init__(
        self,
        input_size,
        d_model,
        num_heads,
        num_layers,
        ff_dim,
        num_classes,
        dropout,
        sequence_length
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_size,
                d_model
            ),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                sequence_length,
                d_model
            )
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.final_norm = nn.LayerNorm(
            d_model
        )

        # Learned attention pooling.
        self.attention_pool = nn.Sequential(
            nn.Linear(
                d_model,
                128
            ),
            nn.Tanh(),
            nn.Linear(
                128,
                1
            )
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                d_model,
                128
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = (
            x
            + self.position_embedding[
                :, :x.shape[1], :
            ]
        )

        x = self.encoder(x)

        x = self.final_norm(x)

        scores = self.attention_pool(
            x
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1
        ).unsqueeze(-1)

        x = (
            x * weights
        ).sum(dim=1)

        return self.classifier(x)


# ============================================================
# DATA
# ============================================================

def load_class_mapping():

    with open(
        CLASS_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_samples(class_mapping):

    samples = []

    for class_name, label in sorted(
        class_mapping.items(),
        key=lambda item: int(item[1])
    ):

        class_dir = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.isdir(
            class_dir
        ):
            continue

        for filename in sorted(
            os.listdir(class_dir)
        ):

            if filename.lower().endswith(
                ".npy"
            ):

                samples.append(
                    (
                        os.path.join(
                            class_dir,
                            filename
                        ),
                        int(label)
                    )
                )

    return samples


def validate_samples(samples):

    valid = []

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

    return valid


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

def load_normalization():

    if not os.path.isfile(
        NORMALIZATION_PATH
    ):
        raise FileNotFoundError(
            "Existing normalization.npz was not found:\n"
            f"{NORMALIZATION_PATH}"
        )

    data = np.load(
        NORMALIZATION_PATH
    )

    mean = data["mean"].astype(
        np.float32
    )

    std = data["std"].astype(
        np.float32
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std
    ).astype(
        np.float32
    )

    if mean.shape != (INPUT_SIZE,):
        raise ValueError(
            f"Normalization mean shape is {mean.shape}, "
            f"expected ({INPUT_SIZE},)"
        )

    return mean, std


# ============================================================
# EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate(
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

        logits = model(
            sequences
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        100.0 * correct / total
        if total
        else 0.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("SignSync TRANSFORMER TRAINER")
    print("=" * 72)

    set_seed(
        RANDOM_SEED
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Dataset: {DATASET_DIR}"
    )

    class_mapping = load_class_mapping()

    if len(class_mapping) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(class_mapping)}"
        )

    samples = load_samples(
        class_mapping
    )

    print(
        f"Discovered sequences: {len(samples)}"
    )

    samples = validate_samples(
        samples
    )

    print(
        f"Valid sequences: {len(samples)}"
    )

    train_samples, val_samples, test_samples = (
        create_splits(samples)
    )

    print(
        f"Train: {len(train_samples)}"
    )

    print(
        f"Validation: {len(val_samples)}"
    )

    print(
        f"Test: {len(test_samples)}"
    )

    mean, std = load_normalization()

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

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": LOADER_WORKERS,
        "pin_memory": DEVICE.type == "cuda"
    }

    if LOADER_WORKERS > 0:
        loader_kwargs[
            "persistent_workers"
        ] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )

    model = SignTransformer(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        sequence_length=SEQUENCE_LENGTH
    ).to(
        DEVICE
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Transformer parameters: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.05
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    best_val = -1.0
    best_epoch = 0

    training_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start = time.time()

        for sequences, labels in train_loader:

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

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP
            ):

                logits = model(
                    sequences
                )

                loss = criterion(
                    logits,
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

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        scheduler.step()

        train_loss = (
            running_loss / total
            if total
            else 0.0
        )

        train_acc = (
            100.0 * correct / total
            if total
            else 0.0
        )

        val_acc = evaluate(
            model,
            val_loader
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {train_loss:.4f} | "
            f"Train {train_acc:.2f}% | "
            f"Val {val_acc:.2f}% | "
            f"LR {scheduler.get_last_lr()[0]:.7f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val:

            best_val = val_acc
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "architecture":
                        "SignTransformer",

                    "input_size":
                        INPUT_SIZE,

                    "sequence_length":
                        SEQUENCE_LENGTH,

                    "num_classes":
                        NUM_CLASSES,

                    "d_model":
                        D_MODEL,

                    "num_heads":
                        NUM_HEADS,

                    "num_layers":
                        NUM_LAYERS,

                    "ff_dim":
                        FF_DIM,

                    "dropout":
                        DROPOUT,

                    "pooling":
                        "learned_attention",

                    "representation":
                        "body_relative"
                },
                MODEL_PATH
            )

    # Load best checkpoint.
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

    final_val = evaluate(
        model,
        val_loader
    )

    test_acc = evaluate(
        model,
        test_loader
    )

    total_time = (
        time.time()
        - training_start
    )

    save_json(
        TRAINING_INFO_PATH,
        {
            "architecture":
                "SignTransformer",

            "representation":
                "body_relative",

            "pooling":
                "learned_attention",

            "sequence_length":
                SEQUENCE_LENGTH,

            "input_size":
                INPUT_SIZE,

            "num_classes":
                NUM_CLASSES,

            "d_model":
                D_MODEL,

            "num_heads":
                NUM_HEADS,

            "num_layers":
                NUM_LAYERS,

            "feedforward_dimension":
                FF_DIM,

            "best_validation_accuracy":
                best_val,

            "best_epoch":
                best_epoch,

            "final_validation_accuracy":
                final_val,

            "test_accuracy":
                test_acc,

            "device":
                str(DEVICE),

            "gpu":
                (
                    torch.cuda.get_device_name(0)
                    if DEVICE.type == "cuda"
                    else None
                ),

            "training_seconds":
                total_time
        }
    )

    print()
    print("=" * 72)
    print("TRANSFORMER TRAINING COMPLETE")
    print("=" * 72)
    print(
        f"Best validation accuracy : "
        f"{best_val:.2f}%"
    )
    print(
        f"Final validation accuracy: "
        f"{final_val:.2f}%"
    )
    print(
        f"Test accuracy            : "
        f"{test_acc:.2f}%"
    )
    print(
        f"Model: {MODEL_PATH}"
    )
    print(
        f"Training info: {TRAINING_INFO_PATH}"
    )


if __name__ == "__main__":
    main()
