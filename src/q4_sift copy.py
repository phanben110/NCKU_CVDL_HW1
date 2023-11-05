import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
def sift(): 
    st.image("image/sift.png")
    st.write("## Camera Calibration")

    value = streamlit_image_coordinates("https://placekitten.com/200/300")

    st.write(value)