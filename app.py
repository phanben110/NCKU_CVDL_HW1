import streamlit as st
from streamlit_option_menu import option_menu
import os           
import cv2 
import numpy as np
from src.q1_camera_calibration import camera_calibration
from src.q2_augmented_reality import augmented_reality  
from src.q3_stereo_disparity_map import stereo_disparity_map
from src.q4_sift import sift
from src.q5_cifar10_vgg19 import cifar10_vgg19

# Create an option menu for the main menu in the sidebar
st.set_page_config(page_title="CV & DL Homework1", page_icon="image/logo_csie2.png")
# st.image("image/title_search.png")
st.sidebar.image("image/logo_NCKU.jpeg", use_column_width=True)
with st.sidebar:
    selected = option_menu("Assignment", ["1. Camera Calibration", "2. Augmented Reality", "3. Stereo Disparity Map", "4. SIFT", "5. Training VGG19"],
                           icons=['camera-fill','headset-vr', 'diagram-2-fill', "border-style", "robot"], menu_icon="bars", default_index=0)
# Based on the selected option, you can display different content in your web application
if selected == "1. Camera Calibration":
    camera_calibration()

elif selected == "2. Augmented Reality":
    augmented_reality()

elif selected == "3. Stereo Disparity Map":
    stereo_disparity_map()

elif selected == "4. SIFT":
    sift()

elif selected == "5. Training VGG19":
    cifar10_vgg19() 