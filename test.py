import argparse
import os
import sys
import time
import urllib.request
import math
import subprocess

import cv2
import mediapipe as mp


# Default location to store the model next to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")

# Public MediaPipe model URL (float16 variant)
# Source: MediaPipe Face Landmarker models bucket
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """Ensure the face landmarker model file exists locally; download if missing."""
    if os.path.exists(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Downloading model to {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Model downloaded.")
    return model_path


def create_landmarker(model_path: str, running_mode):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
        num_faces=2,
        min_face_detection_confidence=0.85,
        min_face_presence_confidence=0.85,
        min_tracking_confidence=0.85                                                                                                                                                                                                                  ,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def draw_face_bbox(frame_bgr, landmarks_norm):
    """Draw a bounding box and center point for the face using all landmarks."""
    height, width = frame_bgr.shape[:2]
    xs = [int(l.x * width) for l in landmarks_norm]
    ys = [int(l.y * height) for l in landmarks_norm]
    x_min, x_max = max(min(xs), 0), min(max(xs), width - 1)
    y_min, y_max = max(min(ys), 0), min(max(ys), height - 1)
    cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 0), -1)
    cv2.putText(
        frame_bgr,
        f"face: ({cx}, {cy})",
        (x_min, max(0, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def draw_mouth_landmarks(frame_bgr, landmarks_norm):
    """Highlight mouth using key lip landmarks without depending on mediapipe.solutions.

    Uses commonly referenced indices in the 468/478-point topology:
    - 61: left mouth corner
    - 291: right mouth corner
    - 13: upper inner lip center
    - 14: lower inner lip center
    Draws points at these indices and connects them as a diamond.
    """
    height, width = frame_bgr.shape[:2]

    key_indices = {"left": 61, "right": 291, "upper": 13, "lower": 14}
    points_xy = {}

    for name, idx in key_indices.items():
        if idx < len(landmarks_norm):
            lmk = landmarks_norm[idx]
            x, y = int(lmk.x * width), int(lmk.y * height)
            points_xy[name] = (x, y)
            cv2.circle(frame_bgr, (x, y), 3, (0, 0, 255), -1)

    if all(k in points_xy for k in ("left", "upper", "right", "lower")):
        lx, ly = points_xy["left"]
        ux, uy = points_xy["upper"]
        rx, ry = points_xy["right"]
        dx, dy = points_xy["lower"]
        cv2.line(frame_bgr, (lx, ly), (ux, uy), (0, 0, 255), 1)
        cv2.line(frame_bgr, (ux, uy), (rx, ry), (0, 0, 255), 1)
        cv2.line(frame_bgr, (rx, ry), (dx, dy), (0, 0, 255), 1)
        cv2.line(frame_bgr, (dx, dy), (lx, ly), (0, 0, 255), 1)
        tx = (min(lx, ux, rx, dx) + max(lx, ux, rx, dx)) // 2
        ty = min(ly, uy, ry, dy) - 8
        cv2.putText(
            frame_bgr,
            "mouth",
            (tx, max(0, ty)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


def apply_digital_zoom(frame_bgr, zoom_factor: float, center_xy=None):
    """Crop-and-resize digital zoom.

    - Keeps output size identical to input
    - zoom_factor > 1.0 crops a central region and upsamples
    - Optional center (x, y) to anchor the zoom; defaults to image center
    """
    try:
        z = float(zoom_factor)
    except Exception:
        z = 1.0
    if z <= 1.0:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    crop_w = max(1, int(width / z))
    crop_h = max(1, int(height / z))

    if center_xy is None:
        cx, cy = width // 2, height // 2
    else:
        cx, cy = int(center_xy[0]), int(center_xy[1])

    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    x1 = max(0, min(x1, width - crop_w))
    y1 = max(0, min(y1, height - crop_h))

    roi = frame_bgr[y1 : y1 + crop_h, x1 : x1 + crop_w]
    if roi.size == 0:
        return frame_bgr
    return cv2.resize(roi, (width, height), interpolation=cv2.INTER_LINEAR)


def annotate_mouth_state(
    frame_bgr,
    landmarks_norm,
    open_threshold,
    smile_threshold,
    open_ratio_override=None,
    smile_ratio_override=None,
    smile_allowed=True,
):
    """Compute and draw MOUTH OPEN and SMILING states.

    - OPEN: vertical gap between upper(13) and lower(14) lips relative to mouth width (61-291).
    - SMILING: mouth width (61-291) relative to face bbox width.
    """
    height, width = frame_bgr.shape[:2]

    # Compute face bounding box for normalization
    xs = [int(l.x * width) for l in landmarks_norm]
    ys = [int(l.y * height) for l in landmarks_norm]
    x_min, x_max = max(min(xs), 0), min(max(xs), width - 1)
    y_min, y_max = max(min(ys), 0), min(max(ys), height - 1)
    face_w = max(1, x_max - x_min)

    # Key mouth indices
    idx_left, idx_right, idx_up, idx_down = 61, 291, 13, 14
    if max(idx_left, idx_right, idx_up, idx_down) >= len(landmarks_norm):
        return  # Not enough landmarks

    l = landmarks_norm[idx_left]
    r = landmarks_norm[idx_right]
    u = landmarks_norm[idx_up]
    d = landmarks_norm[idx_down]

    lx, ly = int(l.x * width), int(l.y * height)
    rx, ry = int(r.x * width), int(r.y * height)
    ux, uy = int(u.x * width), int(u.y * height)
    dx, dy = int(d.x * width), int(d.y * height)

    mouth_w = math.hypot(rx - lx, ry - ly)
    mouth_h = math.hypot(dx - ux, dy - uy)
    if mouth_w <= 1:
        return

    open_ratio = (mouth_h / mouth_w) if open_ratio_override is None else open_ratio_override
    smile_ratio = (mouth_w / float(face_w)) if smile_ratio_override is None else smile_ratio_override

    is_open = open_ratio >= open_threshold
    is_smile = (smile_ratio >= smile_threshold) and smile_allowed

    # Compose label with live ratios for tuning
    label = []
    if is_open:
        label.append("MOUTH OPEN")
    if is_smile:
        label.append("SMILING")
    label_text = " ".join(label) if label else ""

    # Show numeric ratios for debugging
    debug_text = f"open={open_ratio:.2f} smile={smile_ratio:.2f}"

    # Anchor near the top-left of the mouth bbox area
    anchor_x = min(lx, rx, ux, dx)
    anchor_y = min(ly, ry, uy, dy) - 20
    anchor_y = max(10, anchor_y)

    if label_text:
        cv2.putText(
            frame_bgr,
            label_text,
            (anchor_x, anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame_bgr,
        debug_text,
        (anchor_x, anchor_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run_on_image(image_path: str, model_path: str, open_threshold: float, smile_threshold: float):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Failed to read image: {image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    with create_landmarker(model_path, mp.tasks.vision.RunningMode.IMAGE) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print("No faces detected.")
        cv2.imshow("Face + Mouth Landmarks", image_bgr)
        cv2.waitKey(0)
        return

    for face_idx, lm_list in enumerate(result.face_landmarks):
        draw_face_bbox(image_bgr, lm_list)
        draw_mouth_landmarks(image_bgr, lm_list)
        annotate_mouth_state(image_bgr, lm_list, open_threshold, smile_threshold)

    cv2.imshow("Face + Mouth Landmarks", image_bgr)
    cv2.waitKey(0)


def get_mouth_ratios(landmarks_norm, width: int, height: int):
    idx_left, idx_right, idx_up, idx_down = 61, 291, 13, 14
    if max(idx_left, idx_right, idx_up, idx_down) >= len(landmarks_norm):
        return None
    l = landmarks_norm[idx_left]
    r = landmarks_norm[idx_right]
    u = landmarks_norm[idx_up]
    d = landmarks_norm[idx_down]
    lx, ly = int(l.x * width), int(l.y * height)
    rx, ry = int(r.x * width), int(r.y * height)
    ux, uy = int(u.x * width), int(u.y * height)
    dx, dy = int(d.x * width), int(d.y * height)

    mouth_w = math.hypot(rx - lx, ry - ly)
    mouth_h = math.hypot(dx - ux, dy - uy)
    if mouth_w <= 1:
        return None

    xs = [int(lm.x * width) for lm in landmarks_norm]
    x_min, x_max = max(min(xs), 0), min(max(xs), width - 1)
    face_w = max(1, x_max - x_min)

    open_ratio = mouth_h / mouth_w
    smile_ratio = mouth_w / float(face_w)
    return open_ratio, smile_ratio


class GestureController:
    def __init__(
        self,
        open_threshold: float,
        smile_threshold: float,
        open_hysteresis: float,
        smile_hysteresis: float,
        min_open_ms: int,
        min_smile_ms: int,
        cooldown_ms: int,
        smile_block_ms: int,
        sioyek_enabled: bool,
        sioyek_path: str,
        sioyek_prev_cmd: str,
        sioyek_next_cmd: str,
    ):
        self.open_threshold = open_threshold
        self.smile_threshold = smile_threshold
        self.open_hysteresis = open_hysteresis
        self.smile_hysteresis = smile_hysteresis
        self.min_open_ms = min_open_ms
        self.min_smile_ms = min_smile_ms
        self.cooldown_ms = cooldown_ms
        self.smile_block_ms = smile_block_ms
        self.sioyek_enabled = sioyek_enabled
        self.sioyek_path = sioyek_path
        self.sioyek_prev_cmd = sioyek_prev_cmd
        self.sioyek_next_cmd = sioyek_next_cmd

        self.open_state = False
        self.smile_state = False
        self.waiting_close_open = False
        self.waiting_close_smile = False
        self.rise_time_open_ms = 0
        self.rise_time_smile_ms = 0
        self.last_trigger_ms = 0
        self.last_open_seen_ms = -10**9
        self._smile_blocked = False

    def _now_ms(self):
        return int(time.monotonic() * 1000)

    def _cooldown_ready(self, now_ms: int) -> bool:
        return (now_ms - self.last_trigger_ms) >= self.cooldown_ms

    def _run_sioyek(self, cmd: str):
        if not self.sioyek_enabled:
            return
        try:
            subprocess.run([self.sioyek_path, "--execute-command", cmd], check=False)
        except Exception:
            pass

    def update(self, open_ratio: float, smile_ratio: float):
        now_ms = self._now_ms()

        # Hysteresis for open
        if not self.open_state:
            if open_ratio >= self.open_threshold:
                self.open_state = True
        else:
            if open_ratio <= max(0.0, self.open_threshold - self.open_hysteresis):
                self.open_state = False

        # Track last time mouth was open (for smile suppression)
        if self.open_state:
            self.last_open_seen_ms = now_ms

        # Determine smile suppression window
        self._smile_blocked = (now_ms - self.last_open_seen_ms) <= self.smile_block_ms

        # Hysteresis for smile
        if not self._smile_blocked:
            if not self.smile_state:
                if smile_ratio >= self.smile_threshold:
                    self.smile_state = True
            else:
                if smile_ratio <= max(0.0, self.smile_threshold - self.smile_hysteresis):
                    self.smile_state = False
        else:
            # While blocked, do not consider smiling
            self.smile_state = False
            self.waiting_close_smile = False

        # Mouth open-close -> previous page
        if not self.waiting_close_open and self.open_state and self._cooldown_ready(now_ms):
            self.waiting_close_open = True
            self.rise_time_open_ms = now_ms
        elif self.waiting_close_open and not self.open_state:
            dur = now_ms - self.rise_time_open_ms
            if dur >= self.min_open_ms and self._cooldown_ready(now_ms):
                self._run_sioyek(self.sioyek_prev_cmd)
                self.last_trigger_ms = now_ms
            self.waiting_close_open = False

        # Smile unsmile -> next page
        if not self.waiting_close_smile and self.smile_state and self._cooldown_ready(now_ms):
            self.waiting_close_smile = True
            self.rise_time_smile_ms = now_ms
        elif self.waiting_close_smile and not self.smile_state:
            dur = now_ms - self.rise_time_smile_ms
            if dur >= self.min_smile_ms and self._cooldown_ready(now_ms):
                self._run_sioyek(self.sioyek_next_cmd)
                self.last_trigger_ms = now_ms
            self.waiting_close_smile = False


def run_on_video(
    source: int,
    model_path: str,
    open_threshold: float,
    smile_threshold: float,
    open_hysteresis: float,
    smile_hysteresis: float,
    min_open_ms: int,
    min_smile_ms: int,
    cooldown_ms: int,
    smile_block_ms: int,
    sioyek_enabled: bool,
    sioyek_path: str,
    sioyek_prev_cmd: str,
    sioyek_next_cmd: str,
    zoom_factor: float,
):
    cap = cv2.VideoCapture(source)
    W, H = 960,720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    controller = GestureController(
        open_threshold,
        smile_threshold,
        open_hysteresis,
        smile_hysteresis,
        min_open_ms,
        min_smile_ms,
        cooldown_ms,
        smile_block_ms,
        sioyek_enabled,
        sioyek_path,
        sioyek_prev_cmd,
        sioyek_next_cmd,
    )

    with create_landmarker(model_path, mp.tasks.vision.RunningMode.VIDEO) as landmarker:
        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if zoom_factor and zoom_factor > 1.0:
                frame_bgr = apply_digital_zoom(frame_bgr, zoom_factor)

            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int((frame_index / fps) * 1000)
            frame_index += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                for lm_list in result.face_landmarks:
                    draw_face_bbox(frame_bgr, lm_list)
                    draw_mouth_landmarks(frame_bgr, lm_list)
                    ratios = get_mouth_ratios(lm_list, frame_bgr.shape[1], frame_bgr.shape[0])
                    if ratios is not None:
                        open_ratio, smile_ratio = ratios
                        controller.update(open_ratio, smile_ratio)
                        annotate_mouth_state(
                            frame_bgr,
                            lm_list,
                            open_threshold,
                            smile_threshold,
                            open_ratio_override=open_ratio,
                            smile_ratio_override=smile_ratio,
                            smile_allowed=(not controller._smile_blocked),
                        )

            cv2.imshow("Face + Mouth Landmarks", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face position and mouth landmarks using MediaPipe Face Landmarker"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an image file. If omitted, webcam is used.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0). Ignored when --image is set.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to face_landmarker.task (downloaded automatically if missing).",
    )
    parser.add_argument(
        "--open-threshold",
        type=float,
        default=0.195,
        help="Threshold for OPEN: mouth_gap/mouth_width ratio (default: 0.195)",
    )
    parser.add_argument(
        "--smile-threshold",
        type=float,
        default=0.39,
        help="Threshold for SMILING: mouth_width/face_width ratio (default: 0.39)",
    )
    parser.add_argument(
        "--open-hysteresis",
        type=float,
        default=0.03,
        help="Hysteresis subtracted from open threshold to switch off (default: 0.03)",
    )
    parser.add_argument(
        "--smile-hysteresis",
        type=float,
        default=0.025,
        help="Hysteresis subtracted from smile threshold to switch off (default: 0.025)",
    )
    parser.add_argument(
        "--cooldown-ms",
        type=int,
        default=600,
        help="Cooldown after any trigger in milliseconds (default: 600)",
    )
    parser.add_argument(
        "--open-min-ms",
        type=int,
        default=70,
        help="Minimum duration mouth must stay open before close triggers (default: 70)",
    )
    parser.add_argument(
        "--smile-min-ms",
        type=int,
        default=70,
        help="Minimum duration smile must hold before unsmile triggers (default: 70)",
    )
    parser.add_argument(
        "--smile-block-ms",
        type=int,
        default=1000,
        help="Suppress smile detection for this many ms after any mouth open (default: 1000)",
    )
    parser.add_argument(
        "--disable-sioyek",
        action="store_true",
        help="Disable sending commands to Sioyek (enabled by default)",
    )
    parser.add_argument(
        "--sioyek-path",
        type=str,
        default="sioyek",
        help="Path or name of Sioyek executable (default: sioyek)",
    )
    parser.add_argument(
        "--sioyek-prev",
        type=str,
        default="previous_page",
        help="Sioyek command for previous page (default: previous_page)",
    )
    parser.add_argument(
        "--sioyek-next",
        type=str,
        default="next_page",
        help="Sioyek command for next page (default: next_page)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Digital zoom factor (>1.0 crops and enlarges center; default: 1.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = ensure_model(args.model)

    if args.image:
        run_on_image(args.image, model_path, args.open_threshold, args.smile_threshold)
    else:
        run_on_video(
            args.camera,
            model_path,
            args.open_threshold,
            args.smile_threshold,
            args.open_hysteresis,
            args.smile_hysteresis,
            args.open_min_ms,
            args.smile_min_ms,
            args.cooldown_ms,
            args.smile_block_ms,
            (not args.disable_sioyek),
            args.sioyek_path,
            args.sioyek_prev,
            args.sioyek_next,
            args.zoom,
        )


if __name__ == "__main__":
    main()


