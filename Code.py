#CODE FOR THE MODEL(On google colab)


#Install and extract dataset

!pip install kaggle

!mkdir ~/.kaggle

!mv kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download gti-upm/leapgestrecog
!unzip /content/leapgestrecog.zip

#Training and saving the model
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define dataset path (update if needed)
dataset_path = "/content/leapGestRecog"  # Example: "C:/Users/YourName/leapgestrecog"

# Define image properties
IMG_SIZE = 128  # Resize images to 128x128
NUM_CLASSES = 10  # 10 different hand gestures

# Initialize lists for data and labels
data = []
labels = []
subject_folders = sorted(os.listdir(dataset_path))  # Sort subjects (00, 01, ..., 09)

# Mapping gestures to numerical labels
gesture_mapping = {}  # Gesture folder names → Numeric labels
gesture_index = 0

# Loop through subjects
for subject in subject_folders:
    subject_path = os.path.join(dataset_path, subject)

    if os.path.isdir(subject_path):  # Ensure it's a folder
        gesture_folders = sorted(os.listdir(subject_path))  # Sort gesture types

        for gesture in gesture_folders:
            gesture_path = os.path.join(subject_path, gesture)

            if os.path.isdir(gesture_path):  # Ensure it's a folder
                if gesture not in gesture_mapping:
                    gesture_mapping[gesture] = gesture_index
                    gesture_index += 1

                # Read images inside the gesture folder
                for img_name in os.listdir(gesture_path):
                    img_path = os.path.join(gesture_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 128x128
                    data.append(img)
                    labels.append(gesture_mapping[gesture])

# Convert lists to NumPy arrays
data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize
labels = np.array(labels)

print(f"✅ Total images processed: {len(data)}")
print(f"✅ Gesture classes mapped: {gesture_mapping}")

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train, y_test = to_categorical(y_train, NUM_CLASSES), to_categorical(y_test, NUM_CLASSES)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save Model
model.save("gesture_model.keras")
print("✅ Model saved as gesture_model.keras")

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%")

#TEST THE MODEL
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ✅ Update this with your extracted dataset path
dataset_path = "/content/leapGestRecog"  # Update if needed

# ✅ Load the trained model
model_path = "gesture_model.keras"  # Update if needed
model = load_model(model_path)

# ✅ Define image properties
IMG_SIZE = 128  # Resize images to 128x128

# ✅ Gesture mapping (same as in training)
gesture_mapping = {
    "01_palm": 0, "02_l": 1, "03_fist": 2, "04_fist_moved": 3, "05_thumb": 4,
    "06_index": 5, "07_ok": 6, "08_palm_moved": 7, "09_c": 8, "10_down": 9
}
gesture_labels = {v: k for k, v in gesture_mapping.items()}  # Reverse mapping

# ✅ Get list of all images
all_images = []
for subject in sorted(os.listdir(dataset_path)):  # Iterate over subjects (00, 01, ...)
    subject_path = os.path.join(dataset_path, subject)
    if os.path.isdir(subject_path):
        for gesture in sorted(os.listdir(subject_path)):  # Iterate over gestures (01_palm, 02_l, ...)
            gesture_path = os.path.join(subject_path, gesture)
            if os.path.isdir(gesture_path):
                for img_file in os.listdir(gesture_path):
                    img_path = os.path.join(gesture_path, img_file)
                    all_images.append(img_path)

# ✅ Show available images and let user pick one
print("\n📂 Available Images for Testing:")
for i, img in enumerate(all_images[:50]):  # Display first 50 images for selection
    print(f"{i + 1}. {img}")

# ✅ Get user input
choice = int(input("\n🔢 Enter the number of the image you want to test: ")) - 1

# ✅ Validate input
if 0 <= choice < len(all_images):
    test_image_path = all_images[choice]
else:
    print("❌ Invalid selection! Choosing a random image instead.")
    test_image_path = np.random.choice(all_images)

# ✅ Load and preprocess the chosen image
img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
img_reshaped = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for model

# ✅ Make prediction
prediction = model.predict(img_reshaped)
predicted_label = np.argmax(prediction)
predicted_gesture = gesture_labels[predicted_label]

# ✅ Display the selected image with prediction
plt.imshow(img, cmap="gray")
plt.title(f"Predicted Gesture: {predicted_gesture}")
plt.axis("off")
plt.show()

print(f"✅ The model predicted: {predicted_gesture}")
print(f"✅ Confidence Scores: {prediction[0]}")
print(f"✅ Tested image path: {test_image_path}")




