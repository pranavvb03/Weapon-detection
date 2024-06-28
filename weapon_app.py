import streamlit as st
from PIL import Image

import os

import google.generativeai as genai

from dotenv import load_dotenv, dotenv_values  # we can use load_dotenv or dotenv_values both perform the same task
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'superb-vigil-423217-c0-23c693897695.json'
load_dotenv()

# print(os.getenv("MY_SECRET_KEY"))
MY_SECRET_KEY = st.secrets['MY_SECRET_KEY']
genai.configure(api_key=MY_SECRET_KEY)

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 1024,
}


def main():
    st.title("Object Finder 🔍")

    disclaimer_message = """This is a object detector model for detecting weapons in images.So upload an image containing weapons. 🙂"""

    # Hide the disclaimer initially
    st.write("")

    # Show the disclaimer if the button is clicked
    with st.expander("Disclaimer ⚠️", expanded=False):
       st.markdown(disclaimer_message)


    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Choose an image ...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

        # Process the image (example: get image dimensions)
        image = Image.open(uploaded_image)
        width, height = image.size
        st.write("Image Dimensions:", f"{width}x{height}")

        if st.button("Identify the objects"):

            st.success("Detecting...")

            vision_model = genai.GenerativeModel('gemini-pro-vision')
            response = vision_model.generate_content(["Extract the objects in the provided image and output them in a list in alphabetical order",image])


            st.write("The objects detected are \n", response.text)

            st.success("Thanks for visiting 🤩!!")

            st.info("Try uploading more images and play around with object detection models, just click on browse files, enjoy 😄!!!")


if __name__ == "__main__":
    main()
