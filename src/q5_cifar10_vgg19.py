import streamlit as st
from PIL import Image
import os
import torchsummary
from src.utils import *
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np

label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

def cifar10_vgg19(): 
    st.image("image/vgg19.png")

    st.sidebar.header("Load Image For Inference")
    img_infer = np.array([])

    uploaded_file_left = st.sidebar.file_uploader("Load Image inference", type=["png","jpg"], accept_multiple_files=False)
    if uploaded_file_left is not None:
        image_data = uploaded_file_left.read()
        image_array = np.frombuffer(image_data, np.uint8)
        img_infer = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img_infer is not None:
            img_infer = cv2.cvtColor(img_infer, cv2.COLOR_BGR2RGB)
            st.sidebar.image(img_infer, caption="Uploaded Image inference", use_column_width=True)
        else:
            st.sidebar.error(f"Failed to read image: {uploaded_file_left.name}")
    

    image_folder = "Dataset_CvDl_Hw1/Q5_image/Q5_1"
    if st.button("Training model"):
        batch_size = 64
        num_epochs = 40
        learning_rate = 0.01
        save_model_path = "best_model.pth"
        save_figure_path = "path_to_saved_figure.png"
        train_loader, test_loader = load_datasets(batch_size)
        train_accuracy, train_loss, val_accuracy, val_loss = train_vgg19_bn(train_loader, test_loader, num_epochs, learning_rate, save_model_path)
        plot_and_save_figure(train_accuracy, train_loss, val_accuracy, val_loss, save_figure_path)
    
    if st.button("5.1 Show Augmentation Images"):
        augment_and_display_images(image_folder, "HorizontalFlip")
        augment_and_display_images(image_folder, "VerticalFlip")
        augment_and_display_images(image_folder, "Rotation")
    if st.button("5.2 Show Model Structure"): 
        vgg19_bn_structure()
    if st.button("5.3 Show Accuracy and Loss"):
        st.markdown(f'<p style="text-align:center; color:red;">Show Accuracy and Loss After Training</p>', unsafe_allow_html=True) 
        st.image("path_to_saved_figure.png")
    if st.button("5.4 Inference"): 
        if len(img_infer.shape) == 1: 
            st.warning("Please upload image to inference.")    
        else:
            model = Q5_Cifar10(modeTrain=False)
            model.load_model("model/best.pth")
            image_pil = Image.fromarray(cv2.cvtColor(img_infer, cv2.COLOR_BGR2RGB))
            max_index , probability = model.inference(image_pil) 
            plt.bar(label_names, probability)
            plt.xlabel('Classes')
            plt.ylabel('Probability')
            plt.title('Probability Distribution of Model Prediction')
            plt.xticks(rotation=45)
            st.pyplot(plt)

            st.info(f"Predicted class: {label_names[max_index]}", icon="ℹ️")





def load_datasets(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def train_vgg19_bn(train_loader, test_loader, num_epochs, learning_rate, save_model_path):
    vgg19_bn_model = models.vgg19_bn(num_classes=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg19_bn_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg19_bn_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    best_accuracy = 0.0

    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    for epoch in range(num_epochs):
        # Training
        vgg19_bn_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = vgg19_bn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy.append(100 * correct / total)
        train_loss.append(running_loss / len(train_loader))

        # Validation
        vgg19_bn_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader, 0):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = vgg19_bn_model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        val_accuracy.append(accuracy)
        val_loss.append(running_loss / len(test_loader))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(vgg19_bn_model.state_dict(), save_model_path)

    return train_accuracy, train_loss, val_accuracy, val_loss

def plot_and_save_figure(train_accuracy, train_loss, val_accuracy, val_loss, save_figure_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy, label='Train Accuracy', color='blue')
    plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.legend()
    plt.title('Loss vs. Epoch')
    
    plt.savefig(save_figure_path)

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
