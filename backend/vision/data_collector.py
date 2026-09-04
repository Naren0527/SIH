import os
import sys
import time
import json

import cv2
import numpy as np

# Allow this file to import landmark_detector.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_DIR = os.path.dirname(CURRENT_DIR)

if VISION_DIR not in sys.path:
    sys.path.insert(0, VISION_DIR)

from landmark_detector import LandmarkDetector


PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "sequences"
)

SEQUENCE_LENGTH = 60

SIGNS = [
    "HELLO",
    "YES",
    "NO",
    "THANK_YOU",
    "PLEASE",
    "HELP",
    "WATER",
    "FOOD",
    "HOSPITAL",
    "PAIN"
]


def create_data_directories():
    os.makedirs(DATA_DIR, exist_ok=True)

    for sign in SIGNS:
        os.makedirs(
            os.path.join(DATA_DIR, sign),
            exist_ok=True
        )


def extract_landmarks(
    hand_result,
    pose_result,
    face_result
):
    """
    Convert MediaPipe results into one fixed-size
    numerical feature vector.
    """

    landmarks = []

    # --------------------------------------------------
    # LEFT + RIGHT HANDS
    # --------------------------------------------------

    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)

    if hand_result.hand_landmarks:
        for index, hand in enumerate(
            hand_result.hand_landmarks
        ):
            handedness = "Right"

            if (
                hand_result.handedness
                and index < len(hand_result.handedness)
            ):
                handedness = (
                    hand_result.handedness[index][0].category_name
                )

            hand_array = np.array(
                [
                    [landmark.x, landmark.y, landmark.z]
                    for landmark in hand
                ],
                dtype=np.float32
            )

            if handedness.lower() == "left":
                left_hand = hand_array
            else:
                right_hand = hand_array

    landmarks.extend(left_hand.flatten())
    landmarks.extend(right_hand.flatten())

    # --------------------------------------------------
    # POSE
    # --------------------------------------------------

    pose = np.zeros((33, 4), dtype=np.float32)

    if pose_result.pose_landmarks:
        detected_pose = pose_result.pose_landmarks[0]

        for index, landmark in enumerate(
            detected_pose[:33]
        ):
            pose[index] = [
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
                if hasattr(landmark, "visibility")
                else 1.0
            ]

    landmarks.extend(pose.flatten())

    # --------------------------------------------------
    # FACE
    # --------------------------------------------------

    # We don't store the complete 478-point face mesh
    # yet. We keep selected facial landmarks that are
    # useful for expression and orientation.
    face_indices = [
        1,
        33,
        61,
        199,
        263,
        291,
        13,
        14,
        70,
        300
    ]

    face_features = np.zeros(
        (len(face_indices), 3),
        dtype=np.float32
    )

    if face_result.face_landmarks:
        face = face_result.face_landmarks[0]

        for i, landmark_index in enumerate(face_indices):
            if landmark_index < len(face):
                landmark = face[landmark_index]

                face_features[i] = [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]

    landmarks.extend(face_features.flatten())

    return np.array(
        landmarks,
        dtype=np.float32
    )


def save_sequence(
    sequence,
    sign,
    sequence_number
):
    sign_directory = os.path.join(
        DATA_DIR,
        sign
    )

    os.makedirs(
        sign_directory,
        exist_ok=True
    )

    filename = os.path.join(
        sign_directory,
        f"{sign.lower()}_{sequence_number:04d}.npy"
    )

    np.save(
        filename,
        np.array(
            sequence,
            dtype=np.float32
        )
    )

    return filename


def save_metadata(
    sign,
    sequence_number,
    filename
):
    metadata_file = os.path.join(
        DATA_DIR,
        "metadata.json"
    )

    metadata = []

    if os.path.exists(metadata_file):
        try:
            with open(
                metadata_file,
                "r",
                encoding="utf-8"
            ) as file:
                metadata = json.load(file)
        except (json.JSONDecodeError, OSError):
            metadata = []

    metadata.append(
        {
            "sign": sign,
            "sequence_number": sequence_number,
            "file": os.path.relpath(
                filename,
                PROJECT_ROOT
            ),
            "frames": SEQUENCE_LENGTH
        }
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )


def collect_sequence(
    detector,
    camera,
    sign,
    sequence_number
):
    sequence = []

    print()
    print("=" * 60)
    print(f"Preparing to record: {sign}")
    print(f"Sequence: {sequence_number}")
    print("=" * 60)

    # Countdown
    for count in [3, 2, 1]:
        start_time = time.time()

        while time.time() - start_time < 1:
            success, frame = camera.read()

            if not success:
                return False

            frame = cv2.flip(
                frame,
                1
            )

            cv2.putText(
                frame,
                f"GET READY: {count}",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Sign: {sign}",
                (40, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(
                "SignSync Data Collector",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

    # Record sequence
    for frame_number in range(SEQUENCE_LENGTH):
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            return False

        frame = cv2.flip(
            frame,
            1
        )

        timestamp_ms = int(
            time.perf_counter() * 1000
        )

        (
            hand_result,
            pose_result,
            face_result
        ) = detector.process_frame(
            frame,
            timestamp_ms
        )

        frame = detector.draw_landmarks(
            frame,
            hand_result,
            pose_result,
            face_result
        )

        landmarks = extract_landmarks(
            hand_result,
            pose_result,
            face_result
        )

        sequence.append(
            landmarks
        )

        progress = (
            frame_number + 1
        )

        cv2.rectangle(
            frame,
            (40, 180),
            (500, 210),
            (80, 80, 80),
            -1
        )

        cv2.rectangle(
            frame,
            (40, 180),
            (
                40 + int(
                    460 * progress / SEQUENCE_LENGTH
                ),
                210
            ),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            f"RECORDING: {progress}/{SEQUENCE_LENGTH}",
            (40, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Sign: {sign}",
            (40, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Perform the sign naturally",
            (40, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "SignSync Data Collector",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    filename = save_sequence(
        sequence,
        sign,
        sequence_number
    )

    save_metadata(
        sign,
        sequence_number,
        filename
    )

    print(
        f"Saved: {filename}"
    )

    return True


def get_next_sequence_number(sign):
    sign_directory = os.path.join(
        DATA_DIR,
        sign
    )

    existing_files = [
        filename
        for filename in os.listdir(
            sign_directory
        )
        if filename.endswith(".npy")
    ]

    return len(existing_files) + 1


def main():
    create_data_directories()

    detector = LandmarkDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        detector.close()
        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print()
    print("=" * 60)
    print("              SIGNSYNC DATA COLLECTOR")
    print("=" * 60)
    print()
    print("Available signs:")

    for index, sign in enumerate(
        SIGNS,
        start=1
    ):
        print(
            f"{index:2}. {sign}"
        )

    print()
    print("Enter the number of the sign to record.")
    print("Enter Q to quit.")

    try:
        while True:
            choice = input(
                "\nSelect sign: "
            ).strip()

            if choice.lower() == "q":
                break

            if not choice.isdigit():
                print(
                    "Please enter a valid number."
                )
                continue

            sign_index = int(choice) - 1

            if sign_index < 0 or sign_index >= len(SIGNS):
                print(
                    "Invalid sign number."
                )
                continue

            sign = SIGNS[sign_index]

            sequence_number = get_next_sequence_number(
                sign
            )

            print()
            print(
                f"Recording {sign} "
                f"sequence #{sequence_number}"
            )

            print(
                "Perform the selected sign "
                "naturally when recording starts."
            )

            input(
                "Press ENTER when ready..."
            )

            success = collect_sequence(
                detector,
                camera,
                sign,
                sequence_number
            )

            if not success:
                break

            print()
            print(
                f"Successfully recorded "
                f"{sign} #{sequence_number}"
            )

            print(
                "You can record another sequence "
                "or choose another sign."
            )

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("Data collection stopped.")
    print(f"Dataset location: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()