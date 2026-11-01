"""Run the SmartCane detector on an image and speak the navigation cues in Persian.

Usage:
    python demo.py --weights models/yolov8n_0.88_58.pt --source path/to/image.jpg
    python demo.py --weights models/yolov8n_0.88_58.pt --source path/to/images/  --save-annotated
"""

import argparse
import os
import random

import cv2
from persianttsfarsi import PersianTTS
from ultralytics import YOLO

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Where to steer the user, relative to the detected object.
directions_fa = {
    "left": "کمی به چپ بروید",
    "center": "مستقیم بروید",
    "right": "کمی به راست بروید",
}

# The platform edge line is a hazard, not a target, so the advice is inverted:
# the user is told to move away from it rather than towards it.
safety_line_directions_fa = {
    "left": "به خط زرد لبه سکو نزدیک هستید، کمی به راست بروید",
    "center": "به خط زرد لبه سکو نزدیک هستید، کمی عقب بروید",
    "right": "به خط زرد لبه سکو نزدیک هستید، کمی به چپ بروید",
}

class_messages_fa = {
    "shelter": "ایستگاه اتوبوس نزدیک است. {}",
    "crosswalk": "خط عابر پیاده نزدیک است. {}",
    "Escalator": "پله برقی نزدیک است. {}",
    "downstair": "راه پله به سمت پایین نزدیک است. {}",
    "upstair": "راه پله به سمت بالا نزدیک است. {}",
    "green": "چراغ سبز است. عبور کنید.",
    "red": "چراغ قرمز است. توقف کنید.",
}


def resolve_image(source):
    """Return a single image path; a directory yields a random image inside it."""
    if os.path.isdir(source):
        images = [f for f in os.listdir(source) if f.lower().endswith(IMAGE_EXTENSIONS)]
        if not images:
            raise SystemExit(f"No images found in {source}")
        return os.path.join(source, random.choice(images))
    return source


def horizontal_position(box, image_width):
    """Bucket a box into the left, centre or right third of the frame."""
    x1, _, x2, _ = box.xyxy[0].cpu().numpy()
    center_x = (x1 + x2) / 2
    if center_x < image_width / 3:
        return "left"
    if center_x > 2 * image_width / 3:
        return "right"
    return "center"


def navigation_message(class_name, position):
    if class_name in ("green", "red"):
        return class_messages_fa[class_name]
    if class_name == "safety-line":
        return safety_line_directions_fa[position]
    return class_messages_fa[class_name].format(directions_fa[position])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="models/yolov8n_0.88_58.pt",
                        help="path to the trained YOLOv8 weights")
    parser.add_argument("--source", required=True,
                        help="image file, or a directory to pick a random image from")
    parser.add_argument("--output-dir", default="output",
                        help="where the generated .mp3 files are written")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold")
    parser.add_argument("--save-annotated", action="store_true",
                        help="also write the image with boxes drawn on it")
    args = parser.parse_args()

    image_path = resolve_image(args.source)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Image: {image_path}")

    model = YOLO(args.weights)
    results = model(image_path, conf=args.conf)
    image_width = cv2.imread(image_path).shape[1]

    if args.save_annotated:
        annotated_path = os.path.join(args.output_dir, "annotated.jpg")
        cv2.imwrite(annotated_path, results[0].plot())
        print(f"Annotated image: {annotated_path}")

    tts = PersianTTS()
    for index, box in enumerate(results[0].boxes):
        class_name = model.names[int(box.cls.cpu().numpy())]
        message = navigation_message(class_name, horizontal_position(box, image_width))
        audio_path = os.path.join(args.output_dir, f"cue_{index:02d}.mp3")
        tts.synthesize(message, audio_path)
        print(f"[{class_name}] {message} -> {audio_path}")

    if not len(results[0].boxes):
        print("No objects detected.")


if __name__ == "__main__":
    main()
