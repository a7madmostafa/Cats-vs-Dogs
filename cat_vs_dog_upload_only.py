
import streamlit as st

from helper import preprocess_image, load_my_model, predict_img


st.title("Cats and Dogs Classification App")

# Load the model
my_model = load_my_model('inception_model.keras')

# Upload an image
img = st.file_uploader("Upload an image for cat or dog", type=["jpg", "png", "jpeg"])

if img is not None:
    # Preprocess the image
    img_prep = preprocess_image(img)

    # Make prediction
    class_name = predict_img(img_prep, my_model)

    # Display the image with the prediction
    st.image(img, caption=f"Predicted : {class_name}", width=300, use_column_width=True)
   
 
