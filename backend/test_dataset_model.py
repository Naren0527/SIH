import os
import json
import glob

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

DATASET_ROOT = os.path.join(
    BASE_DIR,
    "archive",
    "Video_Dataset",
    "Video_Dataset"
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
            dropout=0.2
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.input_projection(x)

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        return self.classifier(x)


def get_classes():

    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_ROOT}"
        )

    classes = []

    for item in os.scandir(
        DATASET_ROOT
    ):

        if item.is_dir():
            classes.append(
                item.name
            )

    classes.sort(
        key=lambda x: x.lower()
    )

    return classes


def get_break_video():

    break_folder = os.path.join(
        DATASET_ROOT,
        "Break"
    )

    if not os.path.isdir(
        break_folder
    ):
        raise FileNotFoundError(
            f"Break class folder not found:\n"
            f"{break_folder}"
        )

    videos = []

    for extension in [
        "*.mp4",
        "*.MP4",
        "*.avi",
        "*.AVI",
        "*.mov",
        "*.MOV"
    ]:

        videos.extend(
            glob.glob(
                os.path.join(
                    break_folder,
                    extension
                )
            )
        )

    videos.sort()

    if not videos:
        raise FileNotFoundError(
            "No Break videos found."
        )

    return videos[0]


def get_sample_indices(
    total_frames
):

    if total_frames <= 0:
        return np.array(
            [],
            dtype=np.int64
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH
    )

    indices = np.round(
        indices
    ).astype(np.int64)

    return indices


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

            values.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

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

        values.extend([
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility
        ])

    values = np.asarray(
        values,
        dtype=np.float32
    )

    if values.shape[0] == 132:
        pose = values

    return pose


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

    features = np.concatenate([
        left,
        right,
        pose
    ])

    return features.astype(
        np.float32
    )


def process_video(
    video_path
):

    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open:\n{video_path}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Total frames: {total_frames}"
    )

    sample_indices = (
        get_sample_indices(
            total_frames
        )
    )

    sequence = np.zeros(
        (
            SEQUENCE_LENGTH,
            INPUT_SIZE
        ),
        dtype=np.float32
    )

    detector = None

    try:

        detector = LandmarkDetector()

        current_frame = 0
        target = 0

        timestamp_ms = 0

        while (
            target < SEQUENCE_LENGTH
        ):

            success, frame = (
                capture.read()
            )

            if not success:
                break

            if (
                current_frame
                == sample_indices[target]
            ):

                timestamp_ms += 100

                (
                    hand_result,
                    pose_result,
                    _
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

                sequence[target] = (
                    features
                )

                target += 1

            current_frame += 1

        if target == 0:
            raise RuntimeError(
                "No frames processed."
            )

        while (
            target < SEQUENCE_LENGTH
        ):

            sequence[target] = (
                sequence[target - 1]
            )

            target += 1

        return sequence

    finally:

        capture.release()

        if detector is not None:
            detector.close()


def load_normalization():

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
    )

    return mean, std


def main():

    print("=" * 70)
    print("SignSync Dataset Model Diagnostic")
    print("=" * 70)

    classes = get_classes()

    print(
        f"Classes found: {len(classes)}"
    )

    break_video = get_break_video()

    print()
    print("Testing original BREAK dataset video:")
    print(
        os.path.basename(
            break_video
        )
    )
    print()

    mean, std = load_normalization()

    device = torch.device("cpu")

    model = SignBiLSTM(
        INPUT_SIZE,
        len(classes)
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

    print()
    print(
        "Processing dataset video..."
    )

    sequence = process_video(
        break_video
    )

    print(
        f"Sequence shape: "
        f"{sequence.shape}"
    )

    print(
        f"Feature count: "
        f"{sequence.shape[1]}"
    )

    normalized = (
        sequence - mean
    ) / std

    input_tensor = (
        torch.from_numpy(
            normalized
            .astype(np.float32)
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        top_values, top_indices = (
            torch.topk(
                probabilities,
                k=min(
                    10,
                    len(classes)
                )
            )
        )

    print()
    print("=" * 70)
    print("DATASET VIDEO PREDICTION")
    print("=" * 70)

    for rank, (
        value,
        index
    ) in enumerate(
        zip(
            top_values.cpu().numpy(),
            top_indices.cpu().numpy()
        ),
        start=1
    ):

        label = classes[
            int(index)
        ]

        confidence = (
            float(value) * 100
        )

        marker = ""

        if label.lower() == "break":
            marker = "  <-- TRUE LABEL"

        print(
            f"{rank:2d}. "
            f"{label:<25} "
            f"{confidence:8.2f}%"
            f"{marker}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()