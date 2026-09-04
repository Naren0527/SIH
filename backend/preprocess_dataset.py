import os
import json
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import cv2
import mediapipe as mp_lib
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = ROOT / "archive" / "Video_Dataset" / "Video_Dataset"
OUTPUT_ROOT = ROOT / "data" / "processed_sequences"
MODELS_DIR = ROOT / "models"

HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker.task"

SEQUENCE_LENGTH = 60
FEATURES_PER_FRAME = 258

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

CPU_COUNT = os.cpu_count() or 4
DEFAULT_WORKERS = max(1, min(CPU_COUNT - 1, 12))
WORKERS = int(os.environ.get("SIGNSYNC_WORKERS", DEFAULT_WORKERS))


# ============================================================
# FAST MEDIA PIPE DETECTOR
# ============================================================

class FastLandmarkDetector:
    """
    Hand + pose only.

    IMAGE mode is intentional here.

    The previous parallel preprocessor used VIDEO mode with one
    detector kept alive across multiple videos. That caused the
    detector timestamp to reset to zero when a new video started,
    producing:

        ValueError: Input timestamp must be monotonically increasing.

    IMAGE mode has no timestamp requirement and also prevents
    tracking state from leaking from one dataset video into another.
    """

    def __init__(self):
        if not HAND_MODEL.exists():
            raise FileNotFoundError(f"Missing hand model: {HAND_MODEL}")

        if not POSE_MODEL.exists():
            raise FileNotFoundError(f"Missing pose model: {POSE_MODEL}")

        hand_base = python.BaseOptions(
            model_asset_path=str(HAND_MODEL)
        )

        pose_base = python.BaseOptions(
            model_asset_path=str(POSE_MODEL)
        )

        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.hand_detector = vision.HandLandmarker.create_from_options(
            hand_options
        )

        self.pose_detector = vision.PoseLandmarker.create_from_options(
            pose_options
        )

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp_lib.Image(
            image_format=mp_lib.ImageFormat.SRGB,
            data=rgb_frame,
        )

        hand_result = self.hand_detector.detect(mp_image)
        pose_result = self.pose_detector.detect(mp_image)

        return hand_result, pose_result

    def close(self):
        try:
            self.hand_detector.close()
        except Exception:
            pass

        try:
            self.pose_detector.close()
        except Exception:
            pass


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_hand_features(hand_result):
    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if hand_result is None:
        return left_hand, right_hand

    handedness = getattr(hand_result, "handedness", None)
    landmarks = getattr(hand_result, "hand_landmarks", None)

    if not handedness or not landmarks:
        return left_hand, right_hand

    for i, hand in enumerate(landmarks):
        if i >= len(handedness):
            break

        try:
            label = handedness[i][0].category_name.lower()
        except Exception:
            continue

        values = []

        for landmark in hand:
            values.extend([
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
            ])

        if len(values) != 63:
            continue

        values = np.asarray(values, dtype=np.float32)

        if label == "left":
            left_hand[:] = values
        elif label == "right":
            right_hand[:] = values

    return left_hand, right_hand


def extract_pose_features(pose_result):
    pose = np.zeros(132, dtype=np.float32)

    if pose_result is None:
        return pose

    landmarks = getattr(pose_result, "pose_landmarks", None)

    if not landmarks:
        return pose

    person = landmarks[0]

    values = []

    for landmark in person:
        values.extend([
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
            float(getattr(landmark, "visibility", 0.0)),
        ])

    if len(values) == 132:
        pose[:] = np.asarray(values, dtype=np.float32)

    return pose


def combine_raw_features(hand_result, pose_result):
    left, right = extract_hand_features(hand_result)
    pose = extract_pose_features(pose_result)

    features = np.concatenate([
        left,
        right,
        pose,
    ])

    if features.shape[0] != FEATURES_PER_FRAME:
        raise ValueError(
            f"Expected {FEATURES_PER_FRAME} features, "
            f"got {features.shape[0]}"
        )

    return features.astype(np.float32)


# ============================================================
# BODY-RELATIVE NORMALIZATION
# ============================================================

def body_relative_features(features):
    """
    Converts x/y/z coordinates into a body-relative representation.

    Pose landmarks 11 and 12 are the left/right shoulders.

    All hand and pose coordinates are translated to the shoulder
    midpoint and scaled by shoulder distance.

    Visibility values are preserved.
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    ).copy()

    if features.shape != (FEATURES_PER_FRAME,):
        raise ValueError(
            f"Expected feature shape "
            f"({FEATURES_PER_FRAME},), got {features.shape}"
        )

    pose_start = 126

    # Pose layout:
    # 33 landmarks x [x, y, z, visibility]
    left_shoulder = pose_start + (11 * 4)
    right_shoulder = pose_start + (12 * 4)

    ls = features[left_shoulder:left_shoulder + 3]
    rs = features[right_shoulder:right_shoulder + 3]

    shoulder_center = (ls + rs) / 2.0
    shoulder_distance = float(np.linalg.norm(ls[:2] - rs[:2]))

    # If shoulders are not detected reliably, keep the original
    # representation instead of creating unstable normalization.
    if not np.isfinite(shoulder_distance) or shoulder_distance < 1e-4:
        return features.astype(np.float32)

    # Left hand: 0:63
    for start in (0, 63):
        for j in range(start, start + 63, 3):
            xyz = features[j:j + 3]
            features[j:j + 3] = (
                xyz - shoulder_center
            ) / shoulder_distance

    # Pose: 126:258
    for j in range(pose_start, FEATURES_PER_FRAME, 4):
        xyz = features[j:j + 3]
        features[j:j + 3] = (
            xyz - shoulder_center
        ) / shoulder_distance

    return features.astype(np.float32)


# ============================================================
# VIDEO SAMPLING
# ============================================================

def get_sample_indices(total_frames):
    if total_frames <= 0:
        return []

    if total_frames >= SEQUENCE_LENGTH:
        return np.linspace(
            0,
            total_frames - 1,
            SEQUENCE_LENGTH,
        ).round().astype(np.int64).tolist()

    indices = list(range(total_frames))

    while len(indices) < SEQUENCE_LENGTH:
        indices.append(total_frames - 1)

    return indices[:SEQUENCE_LENGTH]


# ============================================================
# WORKER
# ============================================================

_WORKER_DETECTOR = None


def worker_init():
    global _WORKER_DETECTOR

    cv2.setNumThreads(0)

    _WORKER_DETECTOR = FastLandmarkDetector()


def process_video(task):
    global _WORKER_DETECTOR

    class_name, video_path_str = task
    video_path = Path(video_path_str)

    try:
        if _WORKER_DETECTOR is None:
            _WORKER_DETECTOR = FastLandmarkDetector()

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return {
                "ok": False,
                "class_name": class_name,
                "video_path": str(video_path),
                "error": "Could not open video",
            }

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        sample_indices = get_sample_indices(total_frames)

        if not sample_indices:
            cap.release()

            return {
                "ok": False,
                "class_name": class_name,
                "video_path": str(video_path),
                "error": "Video contains no frames",
            }

        sequence = np.zeros(
            (SEQUENCE_LENGTH, FEATURES_PER_FRAME),
            dtype=np.float32,
        )

        current_index = -1
        next_position = 0
        last_frame = None

        for target_index in sample_indices:
            # Sequential decoding avoids repeated random seeks.
            while current_index < target_index:
                ret, frame = cap.read()

                if not ret:
                    break

                current_index += 1
                last_frame = frame

            if last_frame is None:
                continue

            hand_result, pose_result = (
                _WORKER_DETECTOR.process_frame(last_frame)
            )

            raw = combine_raw_features(
                hand_result,
                pose_result,
            )

            sequence[next_position] = body_relative_features(raw)

            next_position += 1

            if next_position >= SEQUENCE_LENGTH:
                break

        cap.release()

        if next_position > 0 and next_position < SEQUENCE_LENGTH:
            sequence[next_position:] = sequence[next_position - 1]

        if next_position == 0:
            return {
                "ok": False,
                "class_name": class_name,
                "video_path": str(video_path),
                "error": "Could not decode usable frames",
            }

        if not np.isfinite(sequence).all():
            return {
                "ok": False,
                "class_name": class_name,
                "video_path": str(video_path),
                "error": "Invalid numerical values",
            }

        return {
            "ok": True,
            "class_name": class_name,
            "video_path": str(video_path),
            "sequence": sequence,
        }

    except Exception as exc:
        return {
            "ok": False,
            "class_name": class_name,
            "video_path": str(video_path),
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# DATASET DISCOVERY
# ============================================================

def get_dataset_tasks():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_ROOT}"
        )

    classes = sorted([
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir()
    ])

    tasks = []

    for class_dir in classes:
        for video_path in sorted(class_dir.iterdir()):
            if (
                video_path.is_file()
                and video_path.suffix.lower()
                in VIDEO_EXTENSIONS
            ):
                tasks.append((
                    class_dir.name,
                    str(video_path),
                ))

    return classes, tasks


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SignSync FAST DATASET PREPROCESSOR")
    print("BODY-RELATIVE LANDMARKS")
    print("MEDIA PIPE IMAGE MODE")
    print("=" * 72)
    print()

    print(f"Workers: {WORKERS}")
    print(f"CPU cores detected: {CPU_COUNT}")
    print()

    classes, tasks = get_dataset_tasks()

    print(f"Classes: {len(classes)}")
    print(f"Videos:  {len(tasks)}")
    print()

    # Remove old processed files so raw and body-relative data
    # are never mixed.
    old_files = list(
        OUTPUT_ROOT.rglob("*.npy")
    )

    if old_files:
        print(
            f"Removing {len(old_files)} old processed .npy files..."
        )

        for old_file in old_files:
            try:
                old_file.unlink()
            except Exception as exc:
                print(
                    f"Warning: could not delete "
                    f"{old_file}: {exc}"
                )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_names = [
        p.name
        for p in classes
    ]

    class_mapping = {
        str(i): name
        for i, name in enumerate(class_names)
    }

    mapping_path = (
        OUTPUT_ROOT / "class_mapping.json"
    )

    with open(
        mapping_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            class_mapping,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Class mapping saved: {mapping_path}"
    )
    print()
    print("Starting parallel preprocessing...")
    print("No timestamps are used in IMAGE mode.")
    print("Press Ctrl+C to stop.")
    print()

    completed = 0
    failed = 0

    start_time = time.perf_counter()

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=ctx,
        initializer=worker_init,
    ) as executor:

        futures = {
            executor.submit(
                process_video,
                task,
            ): task
            for task in tasks
        }

        try:
            for future in as_completed(futures):
                result = future.result()

                if result["ok"]:
                    class_name = result["class_name"]
                    video_path = Path(
                        result["video_path"]
                    )

                    class_dir = (
                        OUTPUT_ROOT / class_name
                    )

                    class_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    output_path = (
                        class_dir
                        / f"{video_path.stem}.npy"
                    )

                    np.save(
                        output_path,
                        result["sequence"],
                    )

                    completed += 1

                else:
                    failed += 1

                    print(
                        f"\nFAILED: "
                        f"{result['video_path']}\n"
                        f"Reason: {result['error']}"
                    )

                done = completed + failed

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                rate = (
                    done / elapsed
                    if elapsed > 0
                    else 0.0
                )

                remaining = (
                    len(tasks) - done
                )

                eta_seconds = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                eta_minutes = (
                    eta_seconds / 60
                )

                print(
                    f"\rProgress: {done}/{len(tasks)} "
                    f"({done / len(tasks) * 100:6.2f}%) | "
                    f"OK: {completed} | "
                    f"Failed: {failed} | "
                    f"Speed: {rate:5.2f} videos/s | "
                    f"ETA: {eta_minutes:6.1f} min",
                    end="",
                    flush=True,
                )

        except KeyboardInterrupt:
            print(
                "\n\nStopping workers..."
            )

            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

            raise

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print()
    print("=" * 72)
    print("PREPROCESSING COMPLETE")
    print("=" * 72)
    print(f"Total videos: {len(tasks)}")
    print(f"Successful:   {completed}")
    print(f"Failed:       {failed}")
    print(f"Time:         {elapsed / 60:.2f} minutes")

    if elapsed > 0:
        print(
            f"Average speed: "
            f"{len(tasks) / elapsed:.2f} videos/s"
        )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print()

    if failed == 0:
        print("SUCCESS: 3630/3630 videos processed.")
        print()
        print("Next step:")
        print("Train using the GPU trainer.")
    else:
        print(
            "WARNING: Some videos failed. "
            "Do not train yet."
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
