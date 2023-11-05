import streamlit as st
from src.utils import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
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
        if st.button("3.1 Stereo Disparity Map"): 
            disparity_f = disparity(grayL, grayR)
            print(disparity_f.shape)
            output_image = process_ouput(disparity_f) 
            st.image(output_image, caption="Disparity", use_column_width=True) 

        if st.button("3.2 Checking Disparity"):
            disparity_f = disparity(grayL, grayR)
            print(disparity_f.shape)
            output_image = process_ouput(disparity_f) 
            map_disparity(imgLeft, imgRight, output_image)