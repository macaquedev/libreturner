import cv2

# Choose a backend if you know it:
# Linux: cv2.CAP_V4L2     Windows: cv2.CAP_MSMF or cv2.CAP_DSHOW     macOS: cv2.CAP_AVFOUNDATION
cap = cv2.VideoCapture(2, cv2.CAP_ANY)

# Using MJPG often unlocks higher resolutions on UVC webcams
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Try high → low until one sticks
for w, h in [(3840,2160), (2560,1440), (1920,1080), (1600,1200), (1280,720), (1024,576), (800,600), (640,480)]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # Optional: try to set FPS too
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Warm up and verify actual size
    for _ in range(3):
        cap.read()
    ok, frame = cap.read()
    if ok:
        hh, ww = frame.shape[:2]
        if (ww, hh) == (w, h):
            print(f"Locked {ww}x{hh} @ ~{cap.get(cv2.CAP_PROP_FPS):.0f}fps")
            break

print("Actual:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
