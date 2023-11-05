from src.utils import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import streamlit as st

def augmented_reality():
    st.image("image/reality.png")
    st.sidebar.header("Load Image")
    images = []
    uploaded_files = st.sidebar.file_uploader("Upload one or more BMP images", type=["bmp"], accept_multiple_files=True)

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            image_data = uploaded_file.read()
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is not None:
                images.append(image)
                st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                st.sidebar.error(f"Failed to read image: {uploaded_file.name}")

    if len(images) == 0:
        st.warning("Please upload BMP images.")
    else:
        # Add text input for the string
        string = st.text_input("Enter a string (up to 6 characters):", "NCKU")

        color_hex = st.color_picker("Select Drawing Color", "#FF5733")


        color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        # Convert RGB to BGR by swapping the R and B channels and converting to integers
        color_rgb = (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))

        if not string.isupper():
            string = string.upper()
        

        q2_objps, q2_imageps = calibration(images)
        char_in_board = [
                [7, 5, 0],  # slot 1
                [4, 5, 0],  # slot 2
                [1, 5, 0],  # slot 3
                [7, 2, 0],  # slot 4
                [4, 2, 0],  # slot 5
                [1, 2, 0]  # slot 6
            ]        

        if st.button("2.1 Show Words on Board"):
            # Your code for showing words on the board based on the string
            fs = cv2.FileStorage("Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_onboard.txt", cv2.FILE_STORAGE_READ)


            for index, image in enumerate(images):
                h, w = image.shape[:2]
                draw_image = image.copy()
                ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(q2_objps, q2_imageps, (w, h), None, None)
                if ret:
                    rvec = np.array(rvecs[index])
                    tvec = np.array(tvecs[index]).reshape(3, 1)
                    for i_char, character in enumerate(string):
                        ch = np.float32(fs.getNode(character).mat())
                        line_list = []
                        for eachline in ch:
                            ach = np.float32([char_in_board[i_char], char_in_board[i_char]])
                            eachline = np.add(eachline, ach)
                            image_points, jac = cv2.projectPoints(eachline, rvec, tvec, intrinsic_mtx, dist)
                            line_list.append(image_points)

                        # Use the selected color for drawing characters on the image
                        draw_image = draw_char(draw_image, line_list, color=color_rgb)

                    cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
                    st.image(draw_image, caption="Result Image", use_column_width=True)

        if st.button("2.2 Show Words Vertically"):
            # Your code for showing words vertically based on the string
            fs = cv2.FileStorage("Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_vertical.txt", cv2.FILE_STORAGE_READ)

            for index, image in enumerate(images):
                h, w = image.shape[:2]
                draw_image = image.copy()
                ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(q2_objps, q2_imageps, (w, h), None, None)
                if ret:
                    rvec = np.array(rvecs[index])
                    tvec = np.array(tvecs[index]).reshape(3, 1)
                    for i_char, character in enumerate(string):
                        ch = np.float32(fs.getNode(character).mat())
                        line_list = []
                        for eachline in ch:
                            ach = np.float32([char_in_board[i_char], char_in_board[i_char]])
                            eachline = np.add(eachline, ach)
                            image_points, jac = cv2.projectPoints(eachline, rvec, tvec, intrinsic_mtx, dist)
                            line_list.append(image_points)

                        # Use the selected color for drawing characters on the image
                        draw_image = draw_char(draw_image, line_list, color=color_rgb)

                    cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
                    st.image(draw_image, caption="Result Image", use_column_width=True)

if __name__ == '__main__':
    augmented_reality()
