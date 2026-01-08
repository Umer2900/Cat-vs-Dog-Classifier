# app.py
import streamlit as st
from PIL import Image
from inference import predict

# Page config
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱🐶", layout="centered")

# ---------------- Streamlit UI ---------------- #

# Big centered title
st.markdown("<h1 style='text-align:center; color:#1e40af; font-size:52px; font-weight:900;'>Cat vs Dog Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555; font-size:20px; margin-bottom:50px;'>AI-Powered Image Recognition • Instantly Tell Cat or Dog 🐱🐶</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image of a cat or dog",
    type=["jpg", "jpeg", "png"],
    help="Choose a clear photo for the best result"
)

# Big full-width primary button
if st.button("🔍 Classify Image", use_container_width=True, type="primary"):
    if not uploaded_file:
        st.error("Please upload an image first!")
    else:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Perfectly centered image using columns (reliable and clean)
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjustable ratios for more/less centering space
        with col2:
            st.image(image, caption="Uploaded Image", width=300)  # Fixed nice size
        
        # Run prediction
        with st.spinner("Analyzing the image..."):
            pred_class, confidence = predict(image)
        
        # Determine result
        label = "Cat 🐱" if pred_class == 0 else "Dog 🐶"
        result_color = "#1e40af" if pred_class == 0 else "#dc2626"  # Blue for Cat, Red for Dog
        
        # Styled result section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:{result_color}; text-align:center;'>Prediction:</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color:#f8fbff; padding:40px; border-radius:18px; border-left:10px solid {result_color}; "
            f"font-size:36px; color:{result_color}; font-weight:700; text-align:center; box-shadow: 0 6px 20px rgba(0,0,0,0.08);'>"
            f"{label}</div>",
            unsafe_allow_html=True
        )
        
        # Confidence
        conf_percent = confidence * 100
        st.markdown(
            f"<p style='text-align:center; font-size:22px; color:#444; margin-top:35px;'>"
            f"Confidence: <strong>{conf_percent:.2f}%</strong></p>",
            unsafe_allow_html=True
        )

# Simple footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#777; font-size:14px;'>Built with ❤️ using Streamlit • Clean Red & Blue Theme</p>", unsafe_allow_html=True)