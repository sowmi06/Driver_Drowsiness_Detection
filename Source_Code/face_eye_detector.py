from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os


# face detector function
def face_detector(root, face_cascade_path):
    image_size = 224
    face_image, face_label = [], []
    for parentPath, subdirs, files in os.walk(root):
        for subdir in subdirs:
            path = parentPath + "/" + subdir
            datafile = os.listdir(path)
            for file_name in datafile:
                imgPath = path + '/' + file_name
                image_array = cv2.imread(imgPath, cv2.IMREAD_COLOR)
                face_cascade = cv2.CascadeClassifier(face_cascade_path)
                # face detector to detect the facial features
                detected_face = face_cascade.detectMultiScale(image=image_array, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in detected_face:
                    img = cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # original face image
                    roi_color = img[y:y + h, x:x + w]
                    # resizing the face image
                    resized_array = cv2.resize(roi_color, (image_size, image_size))
                    face_image.append(resized_array)
                    face_label.append(subdir)
    return face_image, face_label


# eye detector function
def eye_detector(root, eye_cascade_path):
    eye_image, eye_label = [], []
    for parentPath, subdirs, files in os.walk(root):
        for subdir in subdirs:
            path = parentPath + "/" + subdir
            datafile = os.listdir(path)
            for file_name in datafile:
                imgPath = path + '/' + file_name
                img = image.load_img(imgPath, target_size=(224, 224))
                eye_image.append(image.img_to_array(img))
                eye_label.append(subdir)
    return eye_image, eye_label


# Label encoder function for encoding the labels
def label_encoder(y):
    label_encode = LabelEncoder()
    y = label_encode.fit_transform(y)
    return y


# test train function
def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test


# function to create duplicate images form the training dataset for the image_augmentation
def populate_images(x, y):
    images, labels = [], []
    unique_labels = np.unique(y)
    for i in unique_labels:
        y_index = np.where(y == i)
        y_index = y_index[0]
        # take 100 images per class for augmentation
        for j in range(100):
            y_inx = y_index[j]
            x_img = x[y_inx]
            y_values = y[y_inx]
            images.append(x_img)
            labels.append(y_values)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


