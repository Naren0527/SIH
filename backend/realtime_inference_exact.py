import os
import json
import time
import threading
from collections import deque, Counter

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
    "signsync_bilstm.pt"
)

NORMALIZATION_PATH = os.path.join(
    BASE_DIR,
    "models",
    "normalization.npz"
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
        x = self.input_projection(x)
        output, _ = self.lstm(x)
        x = output.mean(dim=1)
        return self.classifier(x)


# ============================================================
# CLASSES
# ============================================================

def load_classes():

    class_mapping_path = os.path.join(
        BASE_DIR,
        "data",
        "processed_sequences",
        "class_mapping.json"
    )

    if not os.path.exists(class_mapping_path):
        class_mapping_path = os.path.join(
            BASE_DIR,
            "data",
            "class_mapping.json"
        )

    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(
            f"class_mapping.json not found:\n"
            f"{class_mapping_path}"
        )

    with open(
        class_mapping_path,
        "r",
        encoding="utf-8"
    ) as file:
        mapping = json.load(file)

    # The preprocessing script creates:
    # {"0": "Bear", "1": "Break", ...}
    #
    # Sort by numeric label, NOT alphabetically.
    return [
        mapping[key]
        for key in sorted(
            mapping.keys(),
            key=lambda x: int(x)
        )
    ]


# ============================================================
# NORMALIZATION
# ============================================================

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
                float(landmark.x),
                float(landmark.y),
                float(landmark.z)
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


def extract_pose_features(pose_result):

    pose = np.zeros(
        132,
        dtype=np.float32
    )

    if pose_result is None:
        return pose

    if not pose_result.pose_landmarks:
        return pose

    landmarks = pose_result.pose_landmarks[0]

    values = []

    for landmark in landmarks:

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

    if values.shape[0] == 132:
        pose = values

    return pose


# ============================================================
# EXACT BODY-RELATIVE TRANSFORMATION
# Matches the successful dataset preprocessing.
# ============================================================

def body_relative_features(
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

    pose_start = 126

    left_shoulder_start = (
        pose_start
        + POSE_LEFT_SHOULDER * 4
    )

    right_shoulder_start = (
        pose_start
        + POSE_RIGHT_SHOULDER * 4
    )

    left_shoulder = pose[
        left_shoulder_start:
        left_shoulder_start + 3
    ]

    right_shoulder = pose[
        right_shoulder_start:
        right_shoulder_start + 3
    ]

    # EXACT same fallback logic as preprocessing.
    if (
        not np.all(np.isfinite(left_shoulder))
        or not np.all(np.isfinite(right_shoulder))
        or np.linalg.norm(left_shoulder) < 1e-7
        or np.linalg.norm(right_shoulder) < 1e-7
    ):
        return left, right, pose

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
        not np.isfinite(shoulder_distance)
        or shoulder_distance < 1e-5
    ):
        return left, right, pose

    # LEFT + RIGHT HANDS
    for hand in (left, right):

        for j in range(
            0,
            63,
            3
        ):

            xyz = hand[
                j:j + 3
            ]

            if np.all(
                np.abs(xyz) < 1e-8
            ):
                continue

            hand[
                j:j + 3
            ] = (
                xyz
                - shoulder_center
            ) / shoulder_distance

    # POSE
    normalized_pose = pose.copy()

    for landmark_index in range(33):

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

    left, right = extract_hand_features(
        hand_result
    )

    pose = extract_pose_features(
        pose_result
    )

    left, right, pose = body_relative_features(
        left,
        right,
        pose
    )

    features = np.concatenate([
        left,
        right,
        pose
    ]).astype(np.float32)

    if features.shape[0] != INPUT_SIZE:
        raise ValueError(
            f"Expected {INPUT_SIZE} features, "
            f"got {features.shape[0]}"
        )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("SignSync REALTIME TEST")
    print("CAMERA-STABLE EXACT BODY-RELATIVE PIPELINE")
    print("=" * 72)

    classes = load_classes()
    mean, std = load_normalization()

    print(f"Classes        : {len(classes)}")
    print(f"Input features : {INPUT_SIZE}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device         : {device}")

    if device.type == "cuda":
        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
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
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # ------------------------------------------------------------
    # CAMERA
    #
    # IMPORTANT:
    # A separate capture thread continuously reads the webcam.
    # MediaPipe can therefore never block the camera driver.
    # ------------------------------------------------------------

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

    time.sleep(0.5)

    # ------------------------------------------------------------
    # SHARED CAMERA STATE
    # ------------------------------------------------------------

    camera_lock = threading.Lock()

    latest_frame = None

    camera_running = True
    camera_error_count = 0

    def camera_reader():

        nonlocal latest_frame
        nonlocal camera_running
        nonlocal camera_error_count

        while camera_running:

            ret, frame = cap.read()

            if ret and frame is not None:

                with camera_lock:

                    latest_frame = frame

                camera_error_count = 0

            else:

                camera_error_count += 1

                time.sleep(0.005)

    camera_thread = threading.Thread(
        target=camera_reader,
        daemon=True
    )

    camera_thread.start()

    # Wait for the first usable frame.
    first_frame_deadline = (
        time.perf_counter() + 5.0
    )

    while (
        latest_frame is None
        and time.perf_counter() < first_frame_deadline
    ):

        time.sleep(0.01)

    with camera_lock:

        if latest_frame is None:

            camera_running = False

            cap.release()

            raise RuntimeError(
                "Camera opened but no usable frame arrived."
            )

    # ------------------------------------------------------------
    # MEDIAPIPE
    #
    # This is kept exactly where the previous working pipeline
    # used it. Nothing is throttled.
    # ------------------------------------------------------------

    detector = LandmarkDetector()

    sequence = deque(
        maxlen=SEQUENCE_LENGTH
    )

    prediction_history = deque(
        maxlen=7
    )

    last_prediction = "Waiting..."
    last_confidence = 0.0
    last_top5 = []

    frame_counter = 0

    # Monotonic timestamp owned by the processing loop.
    # It cannot go backwards even when camera frames arrive faster
    # than MediaPipe can process them.
    timestamp_ms = 0

    print()
    print("Camera started.")
    print()
    print("IMPORTANT:")
    print("  Hold ONE sign for about 2 seconds.")
    print("  Press SPACE before each new sign.")
    print("  Q = quit")
    print()

    try:

        while True:

            # ----------------------------------------------------
            # GET THE MOST RECENT CAMERA FRAME
            # ----------------------------------------------------

            with camera_lock:

                if latest_frame is None:
                    continue

                frame = latest_frame.copy()

            # ----------------------------------------------------
            # MEDIA PIPE + FEATURE EXTRACTION
            #
            # Every processed frame gets landmark detection and
            # landmark drawing, exactly like the original working
            # pipeline.
            # ----------------------------------------------------

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

                # ------------------------------------------------
                # LANDMARK DRAWING
                # ------------------------------------------------

                frame = detector.draw_landmarks(
                    frame,
                    hand_result,
                    pose_result,
                    face_result
                )

                # ------------------------------------------------
                # CLASSIFICATION
                #
                # Do not run the model on every single frame.
                # This is the ONLY part intentionally reduced.
                # ------------------------------------------------

                if (
                    len(sequence) == SEQUENCE_LENGTH
                    and frame_counter % 5 == 0
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
                        .to(device)
                    )

                    with torch.inference_mode():

                        logits = model(
                            input_tensor
                        )

                        probabilities = torch.softmax(
                            logits,
                            dim=1
                        )[0]

                        top_values, top_indices = torch.topk(
                            probabilities,
                            k=min(
                                5,
                                len(classes)
                            )
                        )

                    last_top5 = []

                    for value, index in zip(
                        top_values.detach().cpu().numpy(),
                        top_indices.detach().cpu().numpy()
                    ):

                        label = classes[
                            int(index)
                        ]

                        confidence = (
                            float(value) * 100.0
                        )

                        last_top5.append(
                            (
                                label,
                                confidence
                            )
                        )

                    if last_top5:

                        raw_prediction = (
                            last_top5[0][0]
                        )

                        raw_confidence = (
                            last_top5[0][1]
                        )

                        prediction_history.append(
                            raw_prediction
                        )

                        stable_prediction = Counter(
                            prediction_history
                        ).most_common(1)[0][0]

                        stable_confidences = [
                            confidence
                            for label, confidence
                            in last_top5
                            if label == stable_prediction
                        ]

                        last_prediction = (
                            stable_prediction
                        )

                        if stable_confidences:

                            last_confidence = (
                                sum(
                                    stable_confidences
                                )
                                /
                                len(
                                    stable_confidences
                                )
                            )

                        else:

                            last_confidence = (
                                raw_confidence
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

            # ----------------------------------------------------
            # DISPLAY
            # ----------------------------------------------------

            display_frame = cv2.flip(
                frame,
                1
            )

            cv2.rectangle(
                display_frame,
                (10, 10),
                (500, 350),
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
                0.6,
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
                "TOP 5",
                (25, 193),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            y = 222

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
                    0.48,
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
                "SignSync - Real Time Sign Recognition",
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
                prediction_history.clear()

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
