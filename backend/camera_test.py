import cv2
import time

print("Starting camera test...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit(1)

print("Camera opened.")

for i in range(30):
    ret, frame = cap.read()

    print(
        f"Frame {i + 1}: "
        f"ret={ret}, "
        f"shape={None if frame is None else frame.shape}"
    )

    if ret and frame is not None:
        break

    time.sleep(0.1)

if not ret or frame is None:
    print("ERROR: Camera opened but returned no usable frame.")
    cap.release()
    raise SystemExit(1)

print("Camera is working. Opening window...")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Temporary frame failure. Retrying...")
        time.sleep(0.05)
        continue

    cv2.imshow("SignSync Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Camera test finished.")