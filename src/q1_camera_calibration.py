import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define these variables globally
objpoints = []  # 3D point in real-world space
imgpoints = []  # 2D points in the image plane
width = 11
height = 8
objp = np.zeros((height * width, 3), np.float32)
objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corner_executed = False  # Flag to track if "1.1 Find Corner" has been executed
number_image = 0  # Number of the selected image for extrinsic parameters
dist = None  # Distortion coefficients
mtx = None  # Intrinsic matrix

# Initialize rvecs and tvecs
rvecs = []
tvecs = []

def camera_calibration(): 
    global rvecs, tvecs, dist, mtx  # Make rvecs, tvecs, dist, and mtx global

    st.image("image/camera.png")

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
        if st.button("1.1 Find Corner"):
            global corner_executed
            corner_executed = True  # Set the flag to True
            for image in images:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners)

                    find_corner_image = cv2.drawChessboardCorners(image.copy(), (width, height), corners2, ret)
                    find_corner_image = cv2.cvtColor(find_corner_image, cv2.COLOR_BGR2RGB)
                    st.image(find_corner_image, caption="Corners Found", use_column_width=True)

        if st.button("1.2 Find Intrinsic"):
            if not corner_executed:
                st.warning("Please run '1.1 Find Corner' first.")
            else:
                h, w = images[0].shape[:2]
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
                st.write("Intrinsic matrix:")
                st.write(mtx)
                print("Intrinsic matrix: \n", mtx)

        # Create a form for inputting the image number for extrinsic parameters
        number_image = st.number_input("Select an image (from 1 to {}) for find extrinsic".format(len(images)), min_value=1, max_value=len(images), value=1)

        if st.button("1.3 Find Extrinsic"):
            if not corner_executed:
                st.warning("Please run '1.1 Find Corner' first.")
            elif number_image < 1 or number_image > len(images):
                st.warning("Input error: Please input a valid image number.")
            else:
                rvec = rvecs[number_image - 1]
                tvec = tvecs[number_image - 1]
                tvec = tvec.reshape(3, 1)
                if rvec is not None and tvec is not None:
                    Rotation_matrix = cv2.Rodrigues(rvec)[0]
                    Extrinsic_matrix = np.hstack([Rotation_matrix, tvec])
                    st.write("Extrinsic matrix:")
                    st.write(Extrinsic_matrix)
                    print("Extrinsix: \n", Extrinsic_matrix)

        if st.button("1.4 Find Distortion"):
            if dist is not None:
                st.write("Distortion coefficients:")
                st.write(dist[-1])
                print("Distortion: \n", dist[-1])
            else:
                st.warning("Distortion coefficients have not been calculated yet.")

        if st.button("1.5 Show Result"):
            if len(images) > 0 and dist is not None:
                st.success("Distorted vs. Undistorted Images:")
                for i, image in enumerate(images):
                    h, w = image.shape[:2]
                    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                    dst = cv2.undistort(image, mtx, dist, None, newcameramatrix)
                    x, y, w, h = roi
                    dst = dst[y:y+h, x:x+w]
                    dst = cv2.resize(dst, (image.shape[1], image.shape[0]))

                    # Merge distorted and undistorted images for comparison
                    merged_image = cv2.hconcat([image, dst])
                    st.image(merged_image, caption=f"Distorted vs. Undistorted Image {i + 1}", use_column_width=True)

# Run the Streamlit app
if __name__ == "__main__":
    camera_calibration()
