import cv2 as cv
import glob
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
import json
from torchsummary import summary
from torchvision import datasets, transforms, models

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

def concat_image(src, dst):
    # merge = np.concatenate((src, dst), axis=1)
    merge = cv.hconcat([src, dst])
    return merge

def calibration(images:list, width_board = 11, height_board = 8):
    
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((height_board*width_board, 3), np.float32)
    objp[:,:2] = np.mgrid[0:width_board, 0:height_board].T.reshape(-1, 2)

    # Array to store object points and image points from all the image.
    objpoints = [] #3d point in real world space
    imgpoints = [] #2d points in image plane

    for index, image in enumerate(images):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        ret, corner = cv.findChessboardCorners(gray, (width_board, height_board), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corner)
    return objpoints, imgpoints


def init_feature(name):
    """
    The features include orb, akaza, brisk
    """
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    FLANN_INDEX_LSH    = 6

    # if name == 'sift':
    #     detector = cv.xfeatures2d.SIFT_create()
    #     norm = cv.NORM_L2
    # elif name == 'surf':
    #     detector = cv.xfeatures2d.SURF_create(800)
    #     norm = cv.NORM_L2
    if name == 'orb':
        detector = cv.ORB_create(400)
        norm = cv.NORM_HAMMING
    elif name == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif name == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    if 'flann' in name:
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv.BFMatcher(norm)
    return detector, matcher

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def explore_match(image_1, image_2, keypoint_pair, status = None, H = None):
    h1, w1 = image_1.shape[:2]
    h2, w2 = image_2.shape[:2]
    image_match = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
    image_match[:h1, :w1] = image_1
    image_match[:h2, w1: w1+w2] = image_2

    if H is not None:
        corners = np.float32([[0,0], [w1, 0], [w1,h1], [0,h1] ])
        corners = np.int32(cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv.polylines(image_match, [corners], True, (255,255,255), 6)

    if status is None:
        status = np.ones(len(keypoint_pair), np.bool_)
    point_1, point_2 = [], []

    for kpp in keypoint_pair:
        point_1.append(np.int32(kpp[0].pt))
        point_2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0,255,0)
    red = (0,0,255)
    keypoint_color = (125, 108, 96)

    for (x1, y1), (x2, y2), inlier in zip(point_1, point_2, status):
        if inlier:
            color = green
            cv.circle(image_match, (x1,y1), 2, color, -1)
            cv.circle(image_match, (x2,y2), 2, color, -1)
        else:
            color = red
            r = 2
            thickness = 3
            cv.line(image_match, (x1-r, y1-r), (x1+r, y1+r), color, thickness)
            cv.line(image_match, (x1-r, y1+r), (x1+r, y1-r), color, thickness)
            cv.line(image_match, (x2-r, y2-r), (x2+r, y2+r), color, thickness)
            cv.line(image_match, (x2-r, y2+r), (x2+r, y2-r), color, thickness)
    image_match0 = image_match.copy()
    for (x1, y1), (x2, y2), inlier in zip(point_1, point_2, status):
        if inlier:
            color = green
            cv.line(image_match, (x1,y1), (x2,y2), green)
    return image_match


def convert_tuple_to_int(input_tuple):
    # Check if the input tuple has exactly two elements
    if len(input_tuple) != 2:
        raise ValueError("Input tuple must contain exactly two elements")

    # Convert each element of the input tuple to an integer
    output_tuple = (int(input_tuple[0]), int(input_tuple[1]))

    return output_tuple

def draw_char(img, char_list:list, color=(0,0,255)):
    draw_image = img.copy()
    for line in char_list:
        line = line.reshape(2,2)
        # print(tuple(line[0]), tuple(line[1])) 
        draw_image = cv.line(draw_image, convert_tuple_to_int(tuple(line[0])), convert_tuple_to_int(tuple(line[1])), color, 15, cv.LINE_AA)
    return draw_image    

def disparity(imgL, imgR):
    matcher = cv.StereoBM_create(256,25)
    disparity_f = matcher.compute(imgL, imgR)
    return disparity_f

def process_ouput(disparity):
    cv8uc = cv.normalize(disparity, None, alpha=0,
    beta=255, norm_type=cv.NORM_MINMAX,
    dtype=cv.CV_8UC1)
    return cv8uc


def im_convert(image):
    image = image * (np.array((0.4914, 0.4822, 0.4465)) + np.array((0.2023, 0.1994, 0.2010)))
    image = image.clip(0,1)
    return image

class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Conv1
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),  # Conv2
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool1
            nn.Conv2d(32, 64, 3, padding=1),  # Conv3
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),  # Conv4
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool2
            nn.Conv2d(64, 128, 3, padding=1),  # Conv5
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # Conv6
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # Conv7
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool3
            nn.Conv2d(128, 256, 3, padding=1),  # Conv8
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv9
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv10
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool4
            nn.Conv2d(256, 256, 3, padding=1),  # Conv11
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv12
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv13
            nn.ReLU(True),
            # nn.MaxPool2d(2, 2)  # Pool5 
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 256, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Q5_Cifar10:

    def __init__(self, modeTrain= True, name_classes = label_names):
        self.modeTrain = modeTrain & torch.cuda.is_available()

        self.name_classes = name_classes

        num_classes = len(self.name_classes)

        self.hyperparameters = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "SGD", #SGD or Adam
            "maxepoches": 40,
            "lr_drop": 20,
            "lr_decay": 1e-6,
            "momentum": 0.94
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = VGGNet(num_classes)

        self.model= self.model.to(self.device)

        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
        if not self.modeTrain:
            self.load_model("model/best.pth")

        self.loss_fn = nn.CrossEntropyLoss()


    def load_model(self, path):
        if os.path.exists(path):
            if torch.cuda.is_available():
                checkpoint = torch.load(path)
            else:
                checkpoint = torch.load(path, map_location=torch.device('cpu')) 
            print(checkpoint['epoch'])
            self.model.load_state_dict(checkpoint['model_state_dict'])

        else:
            print("Not Found the model")
            sys.exit()
    
    def inference(self, image, show_image=True):
        # Load and preprocess the image
        # image = Image.open(image_path)
        transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        image = transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension

        if torch.cuda.is_available():
            image = image.cuda()

        with torch.no_grad():
            output = self.model(image)

        _, pred = torch.max(output, dim=1)
        y_pred = pred.cpu().item()
        probability = torch.softmax(output, dim=1).cpu().tolist()[0]

        return y_pred, probability

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