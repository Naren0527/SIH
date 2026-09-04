import os
import json
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from vision.landmark_detector import LandmarkDetector


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "signsync_bilstm.pt"
)

NORMALIZATION_PATH = os.path.join(
    BASE_DIR,
    "models",
    "normalization.npz"
)

TRAINING_INFO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "training_info.json"
)

SEQUENCE_LENGTH = 60
INPUT_SIZE = 258

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12


# ============================================================
# MODEL
# ============================================================

class SignBiLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes
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
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.30
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.input_projection(
            x
        )

        output, _ = self.lstm(
            x
        )

        # IMPORTANT:
        # train_model.py uses mean pooling across ALL 60 frames.
        # Realtime inference must use the exact same operation.
        x = output.mean(
            dim=1
        )

        return self.classifier(
            x
        )


# ============================================================
# CLASSES
# ============================================================

def load_classes():

    class_mapping_path = os.path.join(
        BASE_DIR,
        "data",
        "class_mapping.json"
    )

    if os.path.exists(
        class_mapping_path
    ):

        with open(
            class_mapping_path,
            "r",
            encoding="utf-8"
        ) as file:

            mapping = json.load(
                file
            )

        return [
            name
            for name, _ in sorted(
                mapping.items(),
                key=lambda item: item[1]
            )
        ]

    dataset_root = os.path.join(
        BASE_DIR,
        "archive",
        "Video_Dataset",
        "Video_Dataset"
    )

    if not os.path.exists(
        dataset_root
    ):

        raise FileNotFoundError(
            f"Dataset root not found: "
            f"{dataset_root}"
        )

    classes = []

    for item in os.scandir(
        dataset_root
    ):

        if item.is_dir():
            classes.append(
                item.name
            )

    classes.sort(
        key=lambda x: x.lower()
    )

    return classes


# ============================================================
# NORMALIZATION
# ============================================================

def load_normalization():

    data = np.load(
        NORMALIZATION_PATH
    )

    mean = data[
        "mean"
    ].astype(
        np.float32
    )

    std = data[
        "std"
    ].astype(
        np.float32
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std
    )

    return mean, std


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_hand_features(
    hand_result
):

    left = np.zeros(
        63,
        dtype=np.float32
    )

    right = np.zeros(
        63,
        dtype=np.float32
    )

    if hand_result is None:
        return left, right

    if not hand_result.hand_landmarks:
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

        handedness = (
            hand_result
            .handedness[i][0]
            .category_name
            .lower()
        )

        values = []

        for landmark in hand_landmarks:

            values.extend(
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
            )

        values = np.asarray(
            values,
            dtype=np.float32
        )

        if values.shape[0] != 63:
            continue

        if handedness == "left":
            left = values

        elif handedness == "right":
            right = values

    return left, right


def extract_pose_features(
    pose_result
):

    pose = np.zeros(
        132,
        dtype=np.float32
    )

    if pose_result is None:
        return pose

    if not pose_result.pose_landmarks:
        return pose

    landmarks = (
        pose_result.pose_landmarks[0]
    )

    values = []

    for landmark in landmarks:

        visibility = (
            landmark.visibility
            if landmark.visibility is not None
            else 0.0
        )

        values.extend(
            [
                landmark.x,
                landmark.y,
                landmark.z,
                visibility
            ]
        )

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if values.shape[0] == 132:
        pose = values

    return pose


def normalize_frame_features(
    left,
    right,
    pose
):

    left = left.copy().astype(
        np.float32
    )

    right = right.copy().astype(
        np.float32
    )

    pose = pose.copy().astype(
        np.float32
    )

    if pose.shape[0] != 132:
        return left, right, pose

    left_start = (
        POSE_LEFT_SHOULDER * 4
    )

    right_start = (
        POSE_RIGHT_SHOULDER * 4
    )

    left_visibility = pose[
        left_start + 3
    ]

    right_visibility = pose[
        right_start + 3
    ]

    if (
        left_visibility <= 0.05
        or right_visibility <= 0.05
    ):
        return left, right, pose

    left_shoulder = pose[
        left_start:left_start + 3
    ]

    right_shoulder = pose[
        right_start:right_start + 3
    ]

    shoulder_center = (
        left_shoulder
        + right_shoulder
    ) / 2.0

    shoulder_distance = float(
        np.linalg.norm(
            left_shoulder
            - right_shoulder
        )
    )

    if (
        not np.isfinite(
            shoulder_distance
        )
        or shoulder_distance < 0.03
    ):
        return left, right, pose

    def transform_xyz_array(
        values,
        stride
    ):

        result = values.copy()

        for col in range(
            0,
            result.shape[0],
            stride
        ):

            xyz = result[
                col:col + 3
            ]

            if np.allclose(
                xyz,
                0.0
            ):
                continue

            result[
                col:col + 3
            ] = (
                xyz
                - shoulder_center
            ) / shoulder_distance

        return result

    left = transform_xyz_array(
        left,
        3
    )

    right = transform_xyz_array(
        right,
        3
    )

    normalized_pose = pose.copy()

    for landmark_index in range(
        33
    ):

        start = landmark_index * 4

        xyz = normalized_pose[
            start:start + 3
        ]

        normalized_pose[
            start:start + 3
        ] = (
            xyz
            - shoulder_center
        ) / shoulder_distance

    return (
        left.astype(np.float32),
        right.astype(np.float32),
        normalized_pose.astype(np.float32)
    )


def extract_features(
    hand_result,
    pose_result
):

    left, right = (
        extract_hand_features(
            hand_result
        )
    )

    pose = extract_pose_features(
        pose_result
    )

    (
        left,
        right,
        pose
    ) = normalize_frame_features(
        left,
        right,
        pose
    )

    features = np.concatenate(
        [
            left,
            right,
            pose
        ]
    )

    if features.shape[0] != INPUT_SIZE:

        raise ValueError(
            f"Expected {INPUT_SIZE} features, "
            f"got {features.shape[0]}"
        )

    return features.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SignSync Real-Time Inference")
    print("BODY-RELATIVE + MEAN-POOLED MODEL")
    print("=" * 70)

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found: "
            f"{MODEL_PATH}"
        )

    if not os.path.exists(
        NORMALIZATION_PATH
    ):

        raise FileNotFoundError(
            f"Normalization file not found: "
            f"{NORMALIZATION_PATH}"
        )

    classes = load_classes()

    mean, std = (
        load_normalization()
    )

    print(
        f"Classes: {len(classes)}"
    )

    print(
        f"Input features: {INPUT_SIZE}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    device = torch.device(
        "cpu"
    )

    model = SignBiLSTM(
        input_size=INPUT_SIZE,
        num_classes=len(classes)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    print(
        "Model loaded successfully."
    )

    detector = LandmarkDetector()

    cap = cv2.VideoCapture(
        0
    )

    if not cap.isOpened():

        detector.close()

        raise RuntimeError(
            "Could not open webcam."
        )

    sequence = deque(
        maxlen=SEQUENCE_LENGTH
    )

    last_prediction = "Waiting..."
    last_confidence = 0.0
    last_top5 = []

    print()
    print(
        "Camera started."
    )

    print(
        "Perform a sign in front of the camera."
    )

    print(
        "Press Q to quit."
    )

    print()

    try:

        while True:

            ret, frame = (
                cap.read()
            )

            if not ret:

                print(
                    "Could not read webcam frame."
                )

                break

            # Mirror only the displayed camera image.
            # MediaPipe is run on the mirrored frame, matching
            # the representation expected during live interaction.
            frame = cv2.flip(
                frame,
                1
            )

            timestamp_ms = int(
                time.perf_counter()
                * 1000
            )

            try:

                (
                    hand_result,
                    pose_result,
                    face_result
                ) = detector.process_frame(
                    frame,
                    timestamp_ms
                )

                features = (
                    extract_features(
                        hand_result,
                        pose_result
                    )
                )

                normalized = (
                    features - mean
                ) / std

                normalized = np.nan_to_num(
                    normalized,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                sequence.append(
                    normalized.astype(
                        np.float32
                    )
                )

                if (
                    len(sequence)
                    == SEQUENCE_LENGTH
                ):

                    input_sequence = (
                        np.asarray(
                            sequence,
                            dtype=np.float32
                        )
                    )

                    input_tensor = (
                        torch.from_numpy(
                            input_sequence
                        )
                        .unsqueeze(0)
                        .to(device)
                    )

                    with torch.no_grad():

                        logits = model(
                            input_tensor
                        )

                        probabilities = (
                            torch.softmax(
                                logits,
                                dim=1
                            )[0]
                        )

                        top_values, top_indices = (
                            torch.topk(
                                probabilities,
                                k=min(
                                    5,
                                    len(classes)
                                )
                            )
                        )

                    last_top5 = []

                    for value, index in zip(
                        top_values.cpu().numpy(),
                        top_indices.cpu().numpy()
                    ):

                        label = classes[
                            int(index)
                        ]

                        confidence = (
                            float(value)
                            * 100.0
                        )

                        last_top5.append(
                            (
                                label,
                                confidence
                            )
                        )

                    if last_top5:

                        last_prediction = (
                            last_top5[0][0]
                        )

                        last_confidence = (
                            last_top5[0][1]
                        )

                frame = detector.draw_landmarks(
                    frame,
                    hand_result,
                    pose_result,
                    face_result
                )

            except Exception as error:

                cv2.putText(
                    frame,
                    f"Error: {str(error)[:70]}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # -------------------------------------------------
            # Prediction panel
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (10, 10),
                (450, 305),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                "SignSync",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Prediction: {last_prediction}",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: "
                f"{last_confidence:.2f}%",
                (25, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "TOP 5",
                (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            y = 175

            for rank, (
                label,
                confidence
            ) in enumerate(
                last_top5,
                start=1
            ):

                text = (
                    f"{rank}. {label}: "
                    f"{confidence:.2f}%"
                )

                cv2.putText(
                    frame,
                    text,
                    (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                y += 25

            cv2.imshow(
                "SignSync - Real Time Sign Recognition",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        detector.close()

        print()
        print(
            "Camera stopped."
        )


if __name__ == "__main__":
    main()
