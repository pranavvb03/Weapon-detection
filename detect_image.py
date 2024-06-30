import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

def app():
    st.header('Object Detection using Streamlit')
    st.subheader('Powered by YOLOv8')
    st.write('Welcome!')
    model = YOLO("yolov8s.pt")  # or select yolov8m/l-world.pt
    object_names = list(model.names.values())
    def blur_region(image, x0, y0, x1, y1):
"""Blur a specific region of the image."""
        ROI = image[y0, x0]
        blur = cv2.GaussianBlur(ROI, (51, 51), 0) # Apply Gaussian Blur
        image[y0, x0] = blur
        return image
    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        selected_objects = st.multiselect('Choose objects to detect', object_names, default=['person']) 
        min_confidence = st.slider('Confidence score', 0.0, 1.0)
        submit_button = st.form_submit_button(label='Submit')

    if uploaded_file is not None and submit_button:
        input_image = np.array(Image.open(uploaded_file))
        original_image = input_image.copy()

        with st.spinner('Processing image...'):
            result = model(input_image)
            for detection in result[0].boxes.data:
                x0, y0 = (int(detection[0]), int(detection[1]))
                x1, y1 = (int(detection[2]), int(detection[3]))
                score = round(float(detection[4]), 2)
                cls = int(detection[5])
                object_name = model.names[cls]
                label = f'{object_name} {score}'

                if model.names[cls] in selected_objects and score > min_confidence:
                    cv2.rectangle(input_image, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.putText(input_image, label, (x0, y0 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            detections = result[0].verbose()
            cv2.putText(input_image, detections, (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        st.image(input_image, caption='Processed Image', use_column_width=True)
        if detections:
            st.write("Detected regions:")
            for i, (x0, y0, x1, y1, label) in enumerate(detections):
                if st.button(f"Blur {label}", key=f"blur_{i}"):
                    original_image = blur_region(original_image, x0, y0, x1, y1)
                    st.image(original_image, caption='Blurred Image', use_column_width=True)

if __name__ == "__main__":
    app()
