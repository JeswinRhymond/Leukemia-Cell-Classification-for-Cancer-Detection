import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

# Load the saved model
MODEL_NAME = "Leukemic-model-v3.h5"
MODEL_PATH = "D:/FinalYearProject/Leukemia/SavedModel/" + MODEL_NAME

print("\n\nModel Loading...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"\n✅ Model loaded successfully from: {MODEL_PATH}")

# Define constants
IMG_SIZE = 224  # Ensure this matches the training image size
testing_folder = "D:/FinalYearProject/Leukemia/Testing_Model_Images/"
os.makedirs(testing_folder, exist_ok = True)  # Ensure folder exists

# Allowed file types
allowed_extensions = {".bmp", ".png", ".jpg", ".jpeg"}

# GUI to select image
print("\n\nWaiting for input image...")
Tk().withdraw()  # Hide root Tkinter window
file_path = filedialog.askopenfilename(title = "Select an image for prediction", 
                                       filetypes = [("Image Files", "*.bmp;*.png;*.jpg;*.jpeg")])

if not file_path:
    print("\n⚠️ No file selected. Exiting...")
else:
    file_name = os.path.basename(file_path)
    save_path = os.path.join(testing_folder, file_name)

    # Save uploaded file for testing
    with open(file_path, "rb") as src, open(save_path, "wb") as dst:
        dst.write(src.read())
    print(f"\n✅ Image saved at: {save_path}")

    # Load and preprocess the image
    img = image.load_img(save_path, target_size = (IMG_SIZE, IMG_SIZE))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis = 0)  # Add batch dimension

    # Make prediction
    print("\n\n--------------------\nMaking Prediction...")
    prediction = model.predict(img_array)[0][0]  # Get prediction score

    # Compute probability percentages
    leukemia_prob = round(prediction * 100, 2)
    non_leukemia_prob = round(100 - leukemia_prob, 2)

    flag = False

    # Display result
    print("\n🔍 \"Prediction Result\":")
    if prediction >= 0.5:
        print(f"\n--------------------\n🛑 The uploaded cell image is \"Leukemic\" with {leukemia_prob:.2f}% probability.\n--------------------")
        flag = True
    else:
        print(f"\n--------------------\n✅ The uploaded cell image is \"Non-Leukemic\" with {non_leukemia_prob:.2f}% probability.\n--------------------")
        flag = False

    # Show image
    img_display = cv2.imread(save_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    # plt.title("Uploaded Image for Prediction\nThe image ", ((flag) ? "is Leukemic" : " is Non-Leukemic"))
    plt.title(f"Uploaded Image for Prediction\nThe image is {'Leukemic' if flag else 'Non-Leukemic'} with {leukemia_prob if flag else non_leukemia_prob:.2f}% probability")
    plt.axis("off")
    plt.show()

    # Ask to delete file
    delete_choice = input("\nThe uploaded image is currently saved at \"" + testing_folder + "\"\n🗑  Do you want to delete the uploaded image? (yes/no): ").strip().lower()

    if delete_choice in ["yes", "y", "ye", "1"]:
        os.remove(save_path)
        print("\n✅ Image deleted successfully.")
    else:
        print("\n📂 Image kept in: \"", testing_folder + "\"")

print("\n--------------------\nScript Closed ...\nThank You !!!\n--------------------\n")
