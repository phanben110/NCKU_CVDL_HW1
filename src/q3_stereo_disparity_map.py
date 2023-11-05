import streamlit as st
from src.utils import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates


def stereo_disparity_map(): 
    st.image("image/map.png")
    status_run1 = False
    st.sidebar.header("Load Images Left and Right")
    imgRight = np.array([])
    imgLeft = np.array([])

    uploaded_file_left = st.sidebar.file_uploader("Load Image Left", type=["png"], accept_multiple_files=False)
    if uploaded_file_left is not None:
        image_data = uploaded_file_left.read()
        image_array = np.frombuffer(image_data, np.uint8)
        imgLeft = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if imgLeft is not None:
            imgLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2RGB)
            st.sidebar.image(imgLeft, caption="Uploaded Image Left", use_column_width=True)
        else:
            st.sidebar.error(f"Failed to read image: {uploaded_file_left.name}")

    uploaded_file_right = st.sidebar.file_uploader("Load Image Right", type=["png"], accept_multiple_files=False)
    if uploaded_file_right is not None:
        image_data = uploaded_file_right.read()
        image_array = np.frombuffer(image_data, np.uint8)
        imgRight = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if imgLeft is not None:
            imgRight = cv2.cvtColor(imgRight, cv2.COLOR_BGR2RGB)
            st.sidebar.image(imgRight, caption="Uploaded Image Right", use_column_width=True)
        else:
            st.sidebar.error(f"Failed to read image: {uploaded_file_right.name}")
    
    if len(imgLeft.shape) == 1 and len(imgRight.shape) == 1: 
        st.warning("Please upload image Left and Right.")    
    elif len(imgLeft.shape) == 1:
        st.warning("Please upload image Left.")
    elif len(imgRight.shape) == 1:
        st.warning("Please upload image Right.")
 
    else:
        grayL = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
        if st.checkbox("3.1 Stereo Disparity Map"): 
            disparity_f = disparity(grayL, grayR)
            print(disparity_f.shape)
            output_image = process_ouput(disparity_f) 
            st.image(output_image, caption="Disparity", use_column_width=True) 

        if st.checkbox("3.2 Checking Disparity"):
            disparity_check()


def process_output(disparity):
    cv8uc = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    return cv8uc

def find_corresponding_point_a(imageL, imageR, pointA):
    # Convert the left and right images to grayscale
    grayL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)

    # Compute the disparity map
    stereo = cv2.StereoBM_create(256,25)
    disparity = stereo.compute(grayL, grayR)

    # Use the disparity map to find the corresponding point A' based on point A
    x_A = int(pointA["x"])
    y_A = int(pointA["y"])
    disparity_value = disparity[y_A, x_A]

    # This is a simplified example to find A' based on disparity
    # You may need more sophisticated calibration for accurate results
    x_A_prime = x_A + disparity_value
    y_A_prime = y_A  # Assuming the y-coordinate doesn't change significantly

    return {"x": x_A_prime, "y": y_A_prime}

def disparity_check():

    # Load the imL.png and imR.png images
    imL_path = "Dataset_CvDl_Hw1/Q3_Image/imL.png"
    imR_path = "Dataset_CvDl_Hw1/Q3_Image/imR.png"
    imL = Image.open(imL_path)
    imR = Image.open(imR_path)

    # Resize the images for display
    resized_imL = imL.resize((imL.width // 3, imL.height // 3))
    resized_imR = imR.resize((imR.width // 3, imR.height // 3))

    # Get click coordinates on imL.png
    st.info("Click a pixel in the image above to find its corresponding point on the right.", icon="ℹ️")

    clicked_point_L = streamlit_image_coordinates(resized_imL)

    if clicked_point_L:
        st.write(f"Clicked point A on imL: {clicked_point_L}")

        # Calculate the corresponding point A' on imR using the disparity map
        point_A_prime = find_corresponding_point_a(cv2.imread(imL_path), cv2.imread(imR_path), clicked_point_L)
        st.write(f"Corresponding point A' on imR (using disparity): {point_A_prime}")

        # Draw point A on imL.png and add text
        imL_array = np.array(resized_imL)
        imL_with_point = cv2.circle(imL_array, (int(clicked_point_L["x"]), int(clicked_point_L["y"])), 10, (0, 255, 0), -1)  # Larger green point
        cv2.putText(imL_with_point, 'A', (int(clicked_point_L["x"]) + 20, int(clicked_point_L["y"]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Draw point A' on imR.png and add text
        imR_array = np.array(resized_imR)
        imR_with_point = cv2.circle(imR_array, (int(point_A_prime["x"]), int(point_A_prime["y"])), 10, (0, 0, 255), -1)  # Larger red point
        cv2.putText(imR_with_point, "A'", (int(point_A_prime["x"]) + 20, int(point_A_prime["y"]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Display both images side by side
        col1, col2 = st.columns(2)
        col1.image(imL_with_point, use_column_width=True, caption="Image with Point A (imgL)")
        col2.image(imR_with_point, use_column_width=True, caption="Image with Point A' (imgR)")

if __name__ == '__main__':
    disparity_check()

