import streamlit as st
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import torchsummary
import torchvision.models as models

def cifar10_vgg19(): 
    st.image("image/vgg19.png")
    image_folder = "Dataset_CvDl_Hw1/Q5_image/Q5_1"
    
    if st.button("5.1 Show Augmentation Images"):
        augment_and_display_images(image_folder, "HorizontalFlip")
        augment_and_display_images(image_folder, "VerticalFlip")
        augment_and_display_images(image_folder, "Rotation")
    if st.button("5.2 Show Model Structure"): 
        vgg19_bn_structure()

def vgg19_bn_structure():

    vgg19_bn_model = models.vgg19_bn(num_classes=10)
    summary_text = torchsummary.summary(vgg19_bn_model, (3, 32, 32))
    st.text(summary_text)
    print(summary_text)

def augment_and_display_images(image_folder, augmentation_type):
    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]
    
    if augmentation_type == "Rotation":
        augmentation_transform = transforms.RandomRotation(30)
        st.markdown(f'<p style="text-align:center; color:red;">C. Augmented Images with {augmentation_type}</p>', unsafe_allow_html=True)
    elif augmentation_type == "VerticalFlip":
        augmentation_transform = transforms.RandomVerticalFlip()
        st.markdown(f'<p style="text-align:center; color:red;">B. Augmented Images with {augmentation_type}</p>', unsafe_allow_html=True)
    elif augmentation_type == "HorizontalFlip":
        augmentation_transform = transforms.RandomHorizontalFlip()
        st.markdown(f'<p style="text-align:center; color:red;">A. Augmented Images with {augmentation_type}</p>', unsafe_allow_html=True)
    

    
    # Create a subplot for displaying augmented images
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    
    for i in range(9):
        image_path = os.path.join(image_folder, image_files[i])
        image = Image.open(image_path)
        
        # Apply data augmentation
        augmented_image = augmentation_transform(image)
        
        # Extract the filename from the path and use it as the label
        filename = os.path.basename(image_path)
        
        # Display the augmented image with the filename as the label
        axes[i].imshow(augmented_image)
        axes[i].set_title(filename[:-4])
        axes[i].axis('off')
    
    # Display the augmented images
    st.pyplot(fig)

if __name__ == '__main__':
    cifar10_vgg19()
