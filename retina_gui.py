import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json
import os

# =========================================================
#   CONFIG
# =========================================================
MODEL_PATH = "data/retina_model.keras"         # <-- your final model
IMAGE_SIZE = (260, 260)                   # <-- model input size
CLASS_NAMES = ["cataract", "normal"]      # <-- your two classes


# =========================================================
#   LOAD MODEL
# =========================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


# =========================================================
#   PREDICTION FUNCTION
# =========================================================
def predict_image(img_path):
    """
    Loads image → resizes → preprocesses → predicts using EfficientNet preprocess_input.
    Returns: (label, confidence%, raw probs_array)
    """

    try:
        img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    except:
        return "invalid_file", 0.0, None

    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    preds = model.predict(arr)[0]     # shape: (2,)
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    confidence = float(preds[idx] * 100)

    return label, confidence, preds


# =========================================================
#   GUI SETUP
# =========================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Retina Cataract Detection System")
app.geometry("970x650")


# =========================================================
#   TITLE
# =========================================================
title = ctk.CTkLabel(
    app,
    text="RETINA CATARACT DETECTION SYSTEM",
    font=("Poppins SemiBold", 40)
)
title.pack(pady=20)


# =========================================================
#   MAIN FRAME
# =========================================================
main_frame = ctk.CTkFrame(app, corner_radius=20)
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

left_frame = ctk.CTkFrame(main_frame, corner_radius=20)
left_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

right_frame = ctk.CTkFrame(main_frame, corner_radius=20)
right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)


# =========================================================
#   IMAGE PREVIEW AREA
# =========================================================
preview_title = ctk.CTkLabel(
    left_frame, text="Uploaded Image Preview", font=("Poppins", 22)
)
preview_title.pack(pady=10)

image_label = ctk.CTkLabel(left_frame, text="")
image_label.pack(pady=10)

status_label = ctk.CTkLabel(
    left_frame, text="No image uploaded", font=("Poppins", 18)
)
status_label.pack(pady=10)


# =========================================================
#   RESULT AREA
# =========================================================
result_title = ctk.CTkLabel(
    right_frame,
    text="Prediction (Confidence %)",
    font=("Poppins SemiBold", 30)
)
result_title.pack(pady=10)

result_label = ctk.CTkLabel(
    right_frame,
    text="",
    font=("Poppins", 26)
)
result_label.pack(pady=10)


# =========================================================
#   UPLOAD BUTTON
# =========================================================
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select Retina Image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif"),
                   ("All Files", "*.*")]
    )

    if not file_path:
        return

    # Validate preview loading
    try:
        img = Image.open(file_path)
        img.verify()
        img = Image.open(file_path)
    except Exception:
        result_label.configure(text="❌ Invalid image file", text_color="red")
        status_label.configure(text="Upload a valid retina image", text_color="red")
        return

    # Show preview
    preview = img.resize((450, 450))
    tk_img = ImageTk.PhotoImage(preview)
    image_label.configure(image=tk_img)
    image_label.image = tk_img

    status_label.configure(text="Processing...", text_color="yellow")

    # Predict
    pred, conf, probs = predict_image(file_path)

    if pred == "invalid_file":
        result_label.configure(text="❌ Unsupported image format", text_color="red")
        status_label.configure(text="Try another retina photo", text_color="red")
        return

    # Output
    result_label.configure(
        text=f"Detected: {pred.upper()} ({conf:.2f}%)",
        text_color=("cyan" if pred == "cataract" else "green")
    )

    status_label.configure(text="Prediction complete ✔", text_color="green")


upload_btn = ctk.CTkButton(
    left_frame,
    text="Upload Image",
    font=("Poppins SemiBold", 24),
    height=70,
    width=260,
    corner_radius=20,
    command=upload_image
)
upload_btn.pack(pady=25)


# =========================================================
#   RUN APP
# =========================================================
app.mainloop()
