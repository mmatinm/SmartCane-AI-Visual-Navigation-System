import os
import random
from ultralytics import YOLO
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import cv2
from persianttsfarsi import PersianTTS


# --- Persian direction messages ---
directions_fa = {
    "left": "کمی به چپ بروید",
    "center": "مستقیم بروید",
    "right": "کمی به راست بروید"
}

# --- Special safety-line directions ---
safety_line_directions_fa = {
    "left": "به خط زرد لبه سکو نزدیک هستید، کمی به راست بروید",
    "center": "به خط زرد لبه سکو نزدیک هستید، کمی عقب بروید",
    "right": "به خط زرد لبه سکو نزدیک هستید، کمی به چپ بروید"
}

# --- Class messages in Persian ---
class_messages_fa = {
    "shelter": "ایستگاه اتوبوس نزدیک است. {}",
    "crosswalk": "خط عابر پیاده نزدیک است. {}",
    "Escalator": "پله برقی نزدیک است. {}",
    "downstair": "راه پله به سمت پایین نزدیک است. {}",
    "upstair": "راه پله به سمت بالا نزدیک است. {}",
    "green": "چراغ سبز است. عبور کنید.",
    "red": "چراغ قرمز است. توقف کنید."
}

test_images_path = r"D:\Edu\test\images"    # replace this 

image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
sample_image = os.path.join(test_images_path, random.choice(image_files))
print(f"🖼 Selected image: {sample_image}")

model = YOLO(r"D:\yolov8n_0.88.pt")    # replace this
results = model(sample_image)

plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

img = cv2.imread(sample_image)
img_width = img.shape[1]

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls.cpu().numpy())
        class_name = model.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        obj_center_x = (x1 + x2) / 2

        if obj_center_x < img_width / 3:
            position = "left"
        elif obj_center_x > 2 * img_width / 3:
            position = "right"
        else:
            position = "center"

        if class_name in ["green", "red"]:
            message_fa = class_messages_fa[class_name]

        elif class_name == "safety-line":
            message_fa = safety_line_directions_fa[position]

        else:
            direction_msg = directions_fa[position]
            message_fa = class_messages_fa[class_name].format(direction_msg)

        print(f"🔊 پیام: {message_fa}")

        tts = PersianTTS()
        tts.synthesize(message_fa, "output1.mp3")

        from IPython.display import Audio
        display(Audio("output1.mp3", autoplay=True))


