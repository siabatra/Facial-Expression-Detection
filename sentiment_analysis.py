import cv2
import torch
from transformers import pipeline
from fer import FER
import matplotlib.pyplot as plt

# Load the emotion detector model
emotion_detector = FER(mtcnn=True)

# Load the image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

# Analyze sentiment from facial expressions
def analyze_emotion(image):
    # Detect faces and emotions in the image
    emotions = emotion_detector.detect_emotions(image)
    if not emotions:
        print("No faces detected.")
        return None

    # Extract and display emotions
    for idx, face_data in enumerate(emotions):
        bounding_box = face_data["box"]
        emotion_scores = face_data["emotions"]
        max_emotion = max(emotion_scores, key=emotion_scores.get)

        # Draw bounding box and emotion text
        x, y, w, h = bounding_box
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, max_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        print(f"Face {idx + 1} emotion: {max_emotion} with confidence {emotion_scores[max_emotion]:.2f}")

    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Main function
if __name__ == "__main__":
    image_path = 'path_to_your_image.jpg'  # Specify your image path here
    image = load_image(image_path)
    analyze_emotion(image)
