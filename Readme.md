
# SmartCane: An AI-Based Visual Navigation System for the Visually Impaired

## Project Overview

The **SmartCane** project introduces an innovative AI-based visual navigation system designed to enhance the independence and safety of visually impaired individuals in public transportation environments. By leveraging the **YOLOv8n** object detection model, the system identifies crucial objects in real-time and provides audio feedback in Persian, empowering users to navigate with greater confidence.

This project addresses a critical challenge faced by the visually impaired, offering a modern technological solution that complements traditional assistive tools like white canes and guide dogs.

## Features

* **Real-time Object Detection:** Utilizes a lightweight and accurate YOLOv8n model to identify key objects in public transport settings.

* **Targeted Object Classes:** Detects 8 specific classes vital for navigation:

  * `Shelter` (Bus shelters)

  * `Crosswalk`

  * `Escalator`

  * `Downstair` (Stairs going down)

  * `Upstair` (Stairs going up)

  * `Green` (Green traffic lights)

  * `Red` (Red traffic lights)

  * `Safety-line` (Yellow safety lines at subway platform edges)

* **Intelligent Audio Feedback:** Provides real-time spatial navigation instructions in Persian using the `PersianTTSfarsi` library, guiding the user based on the detected object's position (left, center, or right).

* **Optimized for Edge Devices:** The YOLOv8n model is selected for its balance of accuracy and speed, making it suitable for deployment on low-power hardware like a Raspberry Pi 5.

## Project Structure

The project is structured into three main phases, reflected in the provided code and report:

1. **Data Collection and Cleaning (`cleaning.ipynb`):** Focuses on gathering, annotating, and preprocessing diverse datasets to ensure robust model performance.

2. **Object Detection Model Training (`training.ipynb`):** Involves merging datasets, applying data augmentation, and training the YOLOv8n model.

3. **Text-to-Speech Integration (`demo.py`):** Implements the logic for generating navigation messages and converting them into spoken Persian.

## Dataset

The quality and diversity of the dataset are crucial for the model's performance. Our dataset was meticulously curated from Roboflow and includes images collected under various conditions (different lighting, crowd densities, and pedestrian perspectives).

### Final Dataset

The comprehensive dataset used for training is available via Google Drive:
[SmartCaneFinalDataset (Google Drive)](https://www.google.com/search?q=https://drive.google.com/file/d/17kFBDpmrkDPdElNgwIM9HnmgKWnrjxUy/view%3Fusp%3Dsharing)

### Individual Dataset Sources (Roboflow)

* **Crosswalk:** [Crosswalk-lqgin](https://universe.roboflow.com/object-project/crosswalk-lqgin/dataset/1)

* **Pedestrian Traffic Lights (Red and Green):** [Pedestrian-traffic-lights-final](https://www.google.com/search?q=https://universe.roboflow.com/object-project/pedestrian-traffic-lights-final/dataset/1)

* **Bus Shelter:** [bus-shelter](https://www.google.com/search?q=https://universe.roboflow.com/bus-station-detection/bus-shelter/dataset/1), [bus-stop-v3](https://www.google.com/search?q=https://universe.roboflow.com/transportation-vj24a/bus-stop-v3/dataset/1)

* **Escalator, Upstair, Downstair:** [escalator-stairs](https://www.google.com/search?q=https://universe.roboflow.com/objects-in-public-places/escalator-stairs/dataset/1)

* **Safety Line:** [metro-project](https://www.google.com/search?q=https://universe.roboflow.com/metro-station-objects/metro-project/dataset/1), [platform-edge](https://www.google.com/search?q=https://universe.roboflow.com/platform-edge/platform-edge-lines/dataset/1), [subway-objects](https://www.google.com/search?q=https://universe.roboflow.com/subway-detection/subway-objects/dataset/1)

## Model Training and Performance

The **YOLOv8n** model was chosen for its efficiency and accuracy, making it ideal for real-time applications on embedded systems. The model was trained for 40 epochs on the combined and augmented dataset.

### Trained Model Weight

The pre-trained YOLOv8n model used in this project is available for download:
[Trained YOLOv8n Model (yolov8n_0.88_58.pt)](https://drive.google.com/file/d/1Ary5Vyk3IRgoG5ZydxVJFM9tG5BGhZOX/view?usp=sharing)

### Evaluation Results

The model achieved strong performance on the test set, demonstrating high accuracy across all target classes.

| Class | Images | Instances | Box(P) | R | mAP50 | mAP50-95 | 
 | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| All | 1482 | 1672 | 0.872 | 0.827 | 0.877 | 0.546 | 
| Shelter | 200 | 200 | 0.940 | 0.875 | 0.938 | 0.657 | 
| Crosswalk | 200 | 207 | 0.972 | 0.942 | 0.961 | 0.835 | 
| Escalator | 200 | 339 | 0.806 | 0.846 | 0.861 | 0.556 | 
| Downstair | 200 | 201 | 0.884 | 0.871 | 0.917 | 0.489 | 
| Upstair | 200 | 207 | 0.837 | 0.763 | 0.831 | 0.463 | 
| Green | 200 | 205 | 0.857 | 0.898 | 0.914 | 0.419 | 
| Red | 200 | 206 | 0.860 | 0.801 | 0.865 | 0.386 | 
| Safety-line | 82 | 107 | 0.822 | 0.617 | 0.729 | 0.566 | 

The model exhibits excellent overall **mAP@0.5 of 0.877**, indicating its high accuracy. The processing speed is also efficient: **8.0 ms for pre-processing, 259.9 ms for inference, and 0.6 ms for post-processing per image**, making it highly suitable for real-time applications.

### Confusion Matrix

*Figure 1: Confusion Matrix Plot*

The confusion matrix provides a visual representation of the model's ability to correctly classify each object, with strong diagonal values indicating accurate detections.

## Text-to-Speech (TTS) Integration

The **`demo.py`** script handles the integration of object detection results with a Persian Text-to-Speech system, providing audible navigation cues.

### Navigation Message Logic

The system generates dynamic navigation messages based on the detected object's class and its horizontal position (left, center, right) within the image.

**Example Messages:**

* **General Direction Messages (`directions_fa`):**

  * `Left`: کمی به چپ بروید (Go a little to the left)

  * `Center`: مستقیم بروید (Go straight)

  * `Right`: کمی به راست بروید (Go a little to the right)

* **Specific Safety Line Messages (`safety_line_directions_fa`):**

  * `Left`: به خط زرد لبه سکو نزدیک هستید، کمی به راست بروید (You are close to the yellow platform edge line, go a little to the right)

  * `Center`: به خط زرد لبه سکو نزدیک هستید، کمی عقب بروید (You are close to the yellow platform edge line, step back a little)

  * `Right`: به خط زرد لبه سکو نزدیک هستید، کمی به چپ بروید (You are close to the yellow platform edge line, go a little to the left)

* **Class Introduction Messages (`class_messages_fa`):**

  * `Shelter`: ایستگاه اتوبوس نزدیک است. {} (A bus station is near. {})

  * `Crosswalk`: خط عابر پیاده نزدیک است. {} (A crosswalk is near. {})

  * `Escalator`: پله برقی نزدیک است. {} (An escalator is near. {})

  * `Downstair`: راه پله به سمت پایین نزدیک است. {} (Stairs going down are near. {})

  * `Upstair`: راه پله به سمت بالا نزدیک است. {} (Stairs going up are near. {})

  * `Green`: چراغ سبز است. عبور کنید. (The light is green. Cross.)

  * `Red`: چراغ قرمز است. توقف کنید. (The light is red. Stop.)

### Persian TTS Library

The `PersianTTSfarsi` library was chosen for its reliable performance in converting Persian text to speech, ensuring clear and intelligible audio feedback for users.

## Code Repository Structure

* `tts.py`: Python script for integrating object detection with the Persian Text-to-Speech system.

* `cleaning.ipynb`: Jupyter Notebook detailing the data collection, cleaning, and preprocessing steps.

* `training.ipynb`: Jupyter Notebook for merging datasets, applying data augmentation, and training the YOLOv8n model.

* `SmartCane-AI-Visual-Navigation-System_dataset_and_model_links.txt`: A text file containing the Google Drive links for the dataset and the trained model weights.

## Future Work

We envision several exciting avenues for future development:

1. **Hardware Deployment:** Integrate and test the system on a physical SmartCane device equipped with a Raspberry Pi 5.

2. **Dataset Expansion:** Enhance the dataset with more diverse images and new object classes (e.g., station entrances, specific obstacles) to improve generalization.

3. **Object Tracking:** Implement object tracking for smoother and more stable navigation instructions over time.

4. **Depth Perception:** Integrate depth sensors (e.g., stereo cameras, LiDAR) for improved understanding of object distances and collision avoidance, potentially enabling basic 3D mapping.

5. **Advanced Voice Interaction:** Explore more sophisticated Persian TTS models and implement two-way voice interaction for a more dynamic user experience.

6. **Haptic Feedback:** Incorporate haptic feedback (vibrations) to provide additional non-audio cues about proximity and direction.

7. **Personalization:** Allow users to customize audio message types and detail levels for enhanced usability.
