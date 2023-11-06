import streamlit as st
from src.utils import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from streamlit_image_coordinates import streamlit_image_coordinates

def sift(): 
    st.image("image/sift.png")
    image1 = np.array([])
    image2 = np.array([])
    status_run1 = False

    uploaded_file_1 = st.sidebar.file_uploader("Load Image 1", type=["png","jpg"], accept_multiple_files=False)
    if uploaded_file_1 is not None:
        image_data = uploaded_file_1.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image1 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image1 is not None:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            st.sidebar.image(image1, caption="Uploaded Image 1", use_column_width=True)
        else:
            st.sidebar.error(f"Failed to read image: {uploaded_file_1.name}")

    uploaded_file_2 = st.sidebar.file_uploader("Load Image 2", type=["png","jpg"], accept_multiple_files=False)
    if uploaded_file_2 is not None:
        image_data = uploaded_file_2.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image2 is not None:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            st.sidebar.image(image2, caption="Uploaded Image 2", use_column_width=True)
        else:
            st.sidebar.error(f"Failed to read image: {uploaded_file_2.name}")


    
    if len(image2.shape) == 1 and len(image1.shape) == 1: 
        st.warning("Please upload image 1 and 2.", icon="⚠️")    
    elif len(image1.shape) == 1:
        st.warning("Please upload image 1.",  icon="⚠️")
    elif len(image2.shape) == 1:
        st.warning("Please upload image 2.",  icon="⚠️")
 
    else:
        images = []
        images.append(image1)
        images.append(image2)
        

                
        if st.button("4.1 keypoints"):
            status_run1 = True
            detector, matcher = init_feature("brisk")
            keypoints = []
            descriptors = []
            for image in images:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # keypoint, descriptor = sift.detectAndCompute(gray, None)
                keypoint, descriptor = detector.detectAndCompute(gray, None)
                # keypoint = sift.detect(gray, None)
                keypoints.append(keypoint)
                descriptors.append(descriptor)
                image_sift = cv2.drawKeypoints(image.copy(), keypoint, image.copy())
                image_sift = cv2.cvtColor(image_sift, cv2.COLOR_BGR2RGB)
                st.image(image_sift, caption="Key Point", use_column_width=True) 
        
        if st.button("4.2 Matched Keypoints"):
            if status_run1 == False:
                detector, matcher = init_feature("brisk")
                keypoints = []
                descriptors = []
                for image in images:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # keypoint, descriptor = sift.detectAndCompute(gray, None)
                    keypoint, descriptor = detector.detectAndCompute(gray, None)
                    # keypoint = sift.detect(gray, None)
                    keypoints.append(keypoint)
                    descriptors.append(descriptor)
                    image_sift = cv2.drawKeypoints(image.copy(), keypoint, image.copy())
                    image_sift = cv2.cvtColor(image_sift, cv2.COLOR_BGR2RGB)
                # st.image(image_sift, caption="Key Point", use_column_width=True) 
            desc_1, desc_2 = descriptors[:2]
            key_1, key_2 = keypoints[:2]
            img_1, img_2 = images[:2]

            raw_matches = matcher.knnMatch(desc_2, desc_1, k=2)

            point_2, point_1, keypoint_pairs = filter_matches(key_2, key_1, raw_matches)

            if len(point_1) >=4:
                H, status = cv.findHomography(point_2, point_1, cv.RANSAC, 5.0)
                print('{} / {}  inliers/matched'.format(np.sum(status), len(status)))
            else:
                H, status = None, None
            homography = H

            image_match = explore_match(img_2, img_1, keypoint_pairs, status, H)

            image_match = cv2.cvtColor(image_match, cv2.COLOR_BGR2RGB)
            st.image(image_match, caption="Match Key Point", use_column_width=True) 

            img1 = images[0]
            img2 = images[1]

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            if homography is not None:
                warp_image = cv2.warpPerspective(img2, homography, (w1+w2, max(h1, h2)))
                warp_image[:h1, :w1] = img1
                warp_image = cv2.cvtColor(warp_image, cv2.COLOR_BGR2RGB)
                st.image(warp_image, caption="Warp Image", use_column_width=True) 





