
"""
SignSync Transformer realtime inference.

IMPORTANT:
This version makes realtime preprocessing EXACTLY match the currently
trained dataset preprocessing in preprocess_dataset_fast_fixed_v2.py.

The critical fixes are:
1. Shoulder scale uses X/Y distance only, exactly like training.
2. Missing hand landmarks stay transformed exactly like training.
   Do NOT skip zero hand landmarks.
3. Pose XYZ transformation matches training exactly.
4. Camera thread + MediaPipe pipeline stays intact.
5. Uses probability EMA instead of majority-vote labels.
6. Keeps the best-confidence recent window so continuous signing is
   less likely to be dominated by transition frames.
"""

import os
import json
import time
import threading
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from vision.landmark_detector import LandmarkDetector


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "signsync_transformer.pt"
)

NORMALIZATION_PATH = os.path.join(
    BASE_DIR,
    "models",
    "normalization.npz"
)

CLASS_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_sequences",
    "class_mapping.json"
)

SEQUENCE_LENGTH = 60
INPUT_SIZE = 258

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

# Classify every N processed frames.
PREDICTION_INTERVAL = 5

# Keep recent model outputs.
PROBABILITY_HISTORY_LENGTH = 8

# EMA weight for the newest prediction.
EMA_ALPHA = 0.45

# Recent windows kept for continuous-sign handling.
BEST_WINDOW_HISTORY = 8


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
# CLASS MAPPING
# ============================================================

def load_classes():

    paths = [
        CLASS_MAPPING_PATH,
        os.path.join(
            BASE_DIR,
            "data",
            "class_mapping.json"
        )
    ]

    path = None

    for candidate in paths:
        if os.path.isfile(candidate):
            path = candidate
            break

    if path is None:
        raise FileNotFoundError(
            "class_mapping.json not found."
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        mapping = json.load(f)

    # Format:
    # {"0": "Bear", "1": "Break", ...}
    if all(
        str(key).lstrip("-").isdigit()
        for key in mapping.keys()
    ):
        classes = [
            mapping[key]
            for key in sorted(
                mapping.keys(),
                key=lambda key: int(key)
            )
        ]
    else:
        # Format:
        # {"Bear": 0, "Break": 1, ...}
        classes = [
            name
            for name, label in sorted(
                mapping.items(),
                key=lambda item: int(item[1])
            )
        ]

    if len(classes) != 61:
        raise ValueError(
            f"Expected 61 classes, found {len(classes)}."
        )

    return classes


# ============================================================
# NORMALIZATION
# ============================================================

def load_normalization():

    if not os.path.isfile(
        NORMALIZATION_PATH
    ):
        raise FileNotFoundError(
            f"Normalization file not found:\n"
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
            f"Mean shape {mean.shape}, "
            f"expected ({INPUT_SIZE},)."
        )

    if std.shape != (INPUT_SIZE,):
        raise ValueError(
            f"Std shape {std.shape}, "
            f"expected ({INPUT_SIZE},)."
        )

    return mean, std


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_hand_features(hand_result):

    left = np.zeros(
        63,
        dtype=np.float32
    )

    right = np.zeros(
        63,
        dtype=np.float32
    )

    if (
        hand_result is None
        or not hand_result.hand_landmarks
    ):
        return left, right

    for i, hand_landmarks in enumerate(
        hand_result.hand_landmarks
    ):

        if i >= len(
            hand_result.handedness
        ):
            continue

        if not hand_result.handedness[i]:
            continue

        label = (
            hand_result
            .handedness[i][0]
            .category_name
            .lower()
        )

        values = []

        for landmark in hand_landmarks:
            values.extend([
                float(landmark.x),
                float(landmark.y),
                float(landmark.z)
            ])

        values = np.asarray(
            values,
            dtype=np.float32
        )

        if values.shape != (63,):
            continue

        if label == "left":
            left = values

        elif label == "right":
            right = values

    return left, right


def extract_pose_features(pose_result):

    pose = np.zeros(
        132,
        dtype=np.float32
    )

    if (
        pose_result is None
        or not pose_result.pose_landmarks
    ):
        return pose

    person = pose_result.pose_landmarks[0]

    values = []

    for landmark in person:

        visibility = (
            float(landmark.visibility)
            if landmark.visibility is not None
            else 0.0
        )

        values.extend([
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
            visibility
        ])

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if values.shape == (132,):
        pose = values

    return pose


# ============================================================
# EXACT BODY-RELATIVE TRANSFORMATION
# MATCHES preprocess_dataset_fast_fixed_v2.py
# ============================================================

def body_relative_features(
    features
):

    features = np.asarray(
        features,
        dtype=np.float32
    ).copy()

    if features.shape != (INPUT_SIZE,):
        raise ValueError(
            f"Expected ({INPUT_SIZE},), "
            f"got {features.shape}."
        )

    pose_start = 126

    left_shoulder_start = (
        pose_start
        + POSE_LEFT_SHOULDER * 4
    )

    right_shoulder_start = (
        pose_start
        + POSE_RIGHT_SHOULDER * 4
    )

    left_shoulder = features[
        left_shoulder_start:
        left_shoulder_start + 3
    ]

    right_shoulder = features[
        right_shoulder_start:
        right_shoulder_start + 3
    ]

    shoulder_center = (
        left_shoulder
        + right_shoulder
    ) / 2.0

    # CRITICAL:
    # Training uses ONLY X/Y shoulder distance.
    shoulder_distance = float(
        np.linalg.norm(
            left_shoulder[:2]
            - right_shoulder[:2]
        )
    )

    # EXACT training fallback.
    if (
        not np.isfinite(
            shoulder_distance
        )
        or shoulder_distance < 1e-4
    ):
        return features.astype(
            np.float32
        )

    # CRITICAL:
    # Training transforms ALL hand coordinates,
    # including zero-filled missing hands.
    # Do not skip zeros here.
    for start in (0, 63):

        for j in range(
            start,
            start + 63,
            3
        ):

            xyz = features[
                j:j + 3
            ]

            features[
                j:j + 3
            ] = (
                xyz
                - shoulder_center
            ) / shoulder_distance

    # Pose XYZ.
    # Visibility remains unchanged.
    for j in range(
        pose_start,
        INPUT_SIZE,
        4
    ):

        xyz = features[
            j:j + 3
        ]

        features[
            j:j + 3
        ] = (
            xyz
            - shoulder_center
        ) / shoulder_distance

    return features.astype(
        np.float32
    )


def extract_features(
    hand_result,
    pose_result
):

    left, right = extract_hand_features(
        hand_result
    )

    pose = extract_pose_features(
        pose_result
    )

    raw = np.concatenate([
        left,
        right,
        pose
    ]).astype(
        np.float32
    )

    return body_relative_features(
        raw
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("SignSync TRANSFORMER REALTIME - EXACT DATA MATCH")
    print("=" * 72)

    classes = load_classes()
    mean, std = load_normalization()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Classes        : {len(classes)}"
    )
    print(
        f"Input features : {INPUT_SIZE}"
    )
    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )
    print(
        f"Device         : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

    if not os.path.isfile(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model = SignTransformer(
        input_size=INPUT_SIZE,
        d_model=int(
            checkpoint.get(
                "d_model",
                256
            )
        ),
        num_heads=int(
            checkpoint.get(
                "num_heads",
                8
            )
        ),
        num_layers=int(
            checkpoint.get(
                "num_layers",
                3
            )
        ),
        ff_dim=int(
            checkpoint.get(
                "ff_dim",
                512
            )
        ),
        num_classes=len(classes),
        dropout=float(
            checkpoint.get(
                "dropout",
                0.20
            )
        ),
        sequence_length=SEQUENCE_LENGTH
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)
    model.eval()

    print(
        "Transformer loaded successfully."
    )

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not open webcam."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30
    )

    time.sleep(
        0.5
    )

    camera_lock = threading.Lock()
    latest_frame = None
    camera_running = True

    def camera_reader():

        nonlocal latest_frame
        nonlocal camera_running

        while camera_running:

            ret, frame = cap.read()

            if ret and frame is not None:

                with camera_lock:
                    latest_frame = frame

            else:
                time.sleep(
                    0.005
                )

    camera_thread = threading.Thread(
        target=camera_reader,
        daemon=True
    )

    camera_thread.start()

    deadline = (
        time.perf_counter()
        + 5.0
    )

    while (
        latest_frame is None
        and time.perf_counter() < deadline
    ):
        time.sleep(
            0.01
        )

    if latest_frame is None:

        camera_running = False
        cap.release()

        raise RuntimeError(
            "Camera opened but no usable frame arrived."
        )

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    detector = LandmarkDetector()

    sequence = deque(
        maxlen=SEQUENCE_LENGTH
    )

    probability_history = deque(
        maxlen=PROBABILITY_HISTORY_LENGTH
    )

    confidence_history = deque(
        maxlen=BEST_WINDOW_HISTORY
    )

    ema_probabilities = None

    last_prediction = "Waiting..."
    last_confidence = 0.0
    last_top5 = []

    frame_counter = 0
    timestamp_ms = 0

    print()
    print("Camera started.")
    print()
    print("Hold ONE sign for about 2 seconds.")
    print("Press SPACE before each new sign.")
    print("Q = quit")
    print()

    try:

        while True:

            with camera_lock:

                if latest_frame is None:
                    continue

                frame = latest_frame.copy()

            timestamp_ms += 33

            try:

                (
                    hand_result,
                    pose_result,
                    face_result
                ) = detector.process_frame(
                    frame,
                    timestamp_ms
                )

                features = extract_features(
                    hand_result,
                    pose_result
                )

                normalized = (
                    features - mean
                ) / std

                normalized = np.nan_to_num(
                    normalized,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                ).astype(
                    np.float32
                )

                sequence.append(
                    normalized
                )

                frame = detector.draw_landmarks(
                    frame,
                    hand_result,
                    pose_result,
                    face_result
                )

                # =================================================
                # MODEL
                # =================================================

                if (
                    len(sequence)
                    == SEQUENCE_LENGTH
                    and frame_counter % PREDICTION_INTERVAL == 0
                ):

                    input_sequence = np.asarray(
                        sequence,
                        dtype=np.float32
                    )

                    input_tensor = (
                        torch.from_numpy(
                            input_sequence
                        )
                        .unsqueeze(0)
                        .to(
                            device,
                            non_blocking=True
                        )
                    )

                    with torch.inference_mode():

                        logits = model(
                            input_tensor
                        )

                        probabilities = torch.softmax(
                            logits,
                            dim=1
                        )[0]

                    current_probs = (
                        probabilities
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(
                            np.float32
                        )
                    )

                    probability_history.append(
                        current_probs
                    )

                    # Exponential moving average across complete
                    # probability vectors, not just labels.
                    if ema_probabilities is None:

                        ema_probabilities = (
                            current_probs.copy()
                        )

                    else:

                        ema_probabilities = (
                            EMA_ALPHA
                            * current_probs
                            + (1.0 - EMA_ALPHA)
                            * ema_probabilities
                        )

                    # Current smoothed prediction.
                    smooth_index = int(
                        np.argmax(
                            ema_probabilities
                        )
                    )

                    smooth_confidence = float(
                        ema_probabilities[
                            smooth_index
                        ]
                    )

                    # Keep the strongest recent window.
                    confidence_history.append(
                        (
                            smooth_confidence,
                            smooth_index
                        )
                    )

                    best_confidence, best_index = max(
                        confidence_history,
                        key=lambda item: item[0]
                    )

                    # Prefer the smoothed current result unless a
                    # recent window is clearly stronger.
                    if best_confidence > (
                        smooth_confidence + 0.08
                    ):
                        final_index = int(
                            best_index
                        )
                        final_confidence = float(
                            best_confidence
                        )
                    else:
                        final_index = smooth_index
                        final_confidence = smooth_confidence

                    last_prediction = classes[
                        final_index
                    ]

                    last_confidence = (
                        final_confidence * 100.0
                    )

                    top_values, top_indices = torch.topk(
                        torch.from_numpy(
                            ema_probabilities
                        ),
                        k=min(
                            5,
                            len(classes)
                        )
                    )

                    last_top5 = []

                    for value, index in zip(
                        top_values.numpy(),
                        top_indices.numpy()
                    ):

                        last_top5.append(
                            (
                                classes[
                                    int(index)
                                ],
                                float(value) * 100.0
                            )
                        )

            except Exception as error:

                cv2.putText(
                    frame,
                    f"Processing: {str(error)[:70]}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2
                )

            # ====================================================
            # DISPLAY
            # ====================================================

            display_frame = cv2.flip(
                frame,
                1
            )

            cv2.rectangle(
                display_frame,
                (10, 10),
                (520, 350),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                display_frame,
                "SignSync",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Prediction: {last_prediction}",
                (25, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Confidence: {last_confidence:.2f}%",
                (25, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH}",
                (25, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.putText(
                display_frame,
                f"Device: {device}",
                (25, 163),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1
            )

            cv2.putText(
                display_frame,
                "TRANSFORMER",
                (25, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            y = 220

            for rank, (
                label,
                confidence
            ) in enumerate(
                last_top5,
                start=1
            ):

                cv2.putText(
                    display_frame,
                    f"{rank}. {label}: {confidence:.2f}%",
                    (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.47,
                    (255, 255, 255),
                    1
                )

                y += 23

            cv2.putText(
                display_frame,
                "SPACE = reset | Q = quit",
                (25, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "SignSync - Transformer Real Time",
                display_frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            if key == 32:

                sequence.clear()
                probability_history.clear()
                confidence_history.clear()

                ema_probabilities = None

                last_prediction = "Waiting..."
                last_confidence = 0.0
                last_top5 = []

                print(
                    "Sequence reset. "
                    "Perform the next sign."
                )

            frame_counter += 1

    finally:

        camera_running = False

        if camera_thread.is_alive():
            camera_thread.join(
                timeout=1.0
            )

        cap.release()
        cv2.destroyAllWindows()
        detector.close()

        print()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
