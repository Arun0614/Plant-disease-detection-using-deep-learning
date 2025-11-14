
import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

#  Config 
MODEL_PATH = "/Users/arun/deep learning project 1/plant_mobilenetv2.keras"   # saved model
TRAIN_DIR = "/Users/arun/deep learning project 1/dataset/train"
IMG_SIZE = (224, 224)
TOP_K = 3

st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

#  Helpers & Caching 
@st.cache_resource(show_spinner=False)
def load_trained_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    model = load_model(path)
    return model

@st.cache_data(show_spinner=False)
def load_class_names(train_dir):
    if not os.path.isdir(train_dir):
        return []
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    return classes

def preprocess_pil_image(pil_img, target_size):
    img = pil_img.convert("RGB")
    img = img.resize(target_size)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr

def human_friendly_label(class_name):
    return class_name.replace("___", " - ").replace("_", " ")

def healthy_or_infected(class_name):
    label_lower = class_name.lower().replace("_", " ")
    if "healthy" in label_lower or "normal" in label_lower:
        return "Healthy"
    return "Infected"

# Load model and classes
st.title("🌿 Plant Disease Detector")
st.write("Upload an image of a leaf, and the model will predict the disease and plant health status.")

try:
    model = load_trained_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

class_names = load_class_names(TRAIN_DIR)
if not class_names:
    st.warning("Could not locate class folders in the train directory. Prediction will still attempt, "
               "but class labels won't be available.")
else:
    st.info(f" Model loaded successfully. {len(class_names)} classes available.")

#  File uploader UI 
st.markdown("### 📸 Upload a leaf image")
uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("#### Preview")
    placeholder = st.empty()
with col2:
    st.markdown("#### Prediction")
    result_box = st.empty()

if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Unable to open image: {e}")
        st.stop()

    # Display preview image
    placeholder.image(pil_img, use_container_width=True)

    if st.button("Predict"):
        with st.spinner("🧠 Running prediction..."):
            x = preprocess_pil_image(pil_img, IMG_SIZE)
            preds = model.predict(x)[0]

            if class_names:
                top_idx = preds.argsort()[-TOP_K:][::-1]
                results = [(class_names[i], float(preds[i])) for i in top_idx]

                # Display top K predictions
                md = []
                for name, prob in results:
                    nice = human_friendly_label(name)
                    status = healthy_or_infected(name)
                    md.append(f"**{nice}** — {prob*100:.2f}% — *{status}*")
                result_box.markdown("\n\n".join(md))

                # Top prediction details
                top_name, top_prob = results[0]
                top_nice = human_friendly_label(top_name)
                top_status = healthy_or_infected(top_name)

                # 🌿 Dynamic color-coded result box
                if top_status == "Healthy":
                    status_color = "#2ecc71"  # green
                else:
                    status_color = "#e74c3c"  # red

                st.markdown(
                    f"<div style='background-color:{status_color};padding:12px;border-radius:10px;"
                    f"color:white;font-size:18px;text-align:center;'>"
                    f"Top prediction: <b>{top_nice}</b> ({top_prob*100:.2f}% confidence) — <i>{top_status}</i>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Bar chart of top predictions
                top_labels = [human_friendly_label(class_names[i]) for i in top_idx]
                top_probs = [float(preds[i]) for i in top_idx]
                st.bar_chart({lbl: p for lbl, p in zip(top_labels, top_probs)})

            else:
                # No class names
                top_idx = int(np.argmax(preds))
                confidence = float(np.max(preds))
                result_box.write(f"Predicted class index: {top_idx}, confidence: {confidence*100:.2f}%")
                st.success("Prediction returned (no class labels available).")
else:
    placeholder.info("No image uploaded yet — upload a leaf photo to test the model.")

# Footer
st.markdown("---")
st.caption("Model: MobileNetV2 fine-tuned | Developed by Arun 🌱 | Works offline once model is trained.")
