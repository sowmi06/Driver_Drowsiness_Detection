from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os


def face_detector(root, face_cascade_path):
    """A function to read the images from root directory and detect faces from images.
           Parameters
           ----------
           root : root directory path, mandatory
               The root specifies the path of the file containing images.

           face_cascade_path : face cascade classifier directory, mandatory
               The face_cascade_path specifies path of the Harr-Cascade classifier path to detect the faces.

           Steps
           ----------
           STEP 1 : Reads each image in "RBG" format from the specified root path directory.
           STEP 2 : Implements the in-built face detector from the the Haar-Classifier.
           STEP 3 : Fetches 4 coordinate points and constructs the rectangle box around detected faces
           STEP 4 : Resize the image to 224 * 224.
           STEP 5 : Appends both the detected faces and its respective folder name as class_labels.

           Returns
           ----------
           list
                two lists: one list of extracted face images (array) and one list of its respective class labels.
           """

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


def eye_detector(root, eye_cascade_path):
    """A Function to read the images from root directory and detect eyes from images.
           Parameters
           ----------
           root : root directory path, mandatory
               The root specifies the path of the file containing images.

           eye_cascade_path : eye cascade classifier directory, mandatory
               The eye_cascade_path specifies path of the Harr-Cascade classifier path to detect the eyes.

           Steps
           ----------
           STEP 1 : Reads each image in "RBG" format from the specified root path directory.
           STEP 2 : Fetches eye image and constructs the rectangle box around detected faces
           STEP 3 : Resize the image to 224 * 224.
           STEP 4 : Appends both the detected eyes and its respective folder name as class_labels.

           Returns
           ----------
           list
                two lists: one list of extracted eye images (array) and one list of its respective class labels.
           """

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


def label_encoder(y):
    """A Function convert the string labels to numeric labels.
          Parameters
          ----------
          y : y is the class label that needs to be encoded to numeric values.

          Steps
          ----------
          STEP 1 : Creates an object with in-built LabelEncoder() function.
          STEP 2 : Transforms the sting labels to numeric labels using in-buit fit_transform() function.

          Returns
          ----------
          y
               A ndarray with the numeric labels.
          """

    label_encode = LabelEncoder()
    y = label_encode.fit_transform(y)

    return y


def split(x, y):
    """A Function to make a 70 : 30 train test split .
          Parameters
          ----------
          x : x is the total image dataset.
          y : y is the total class label.

          Steps
          ----------
          STEP 1 : Uses an in-built train_test_split() function to split the whole images into 70 : 30.

          Returns
          ----------
          x_train, x_test, y_train, y_test
               x_train and y_train with 70 % of the data.
               x_test and y_test with 30 % of the data.
          """

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, shuffle=True)

    return x_train, x_test, y_train, y_test


def populate_images(x, y):
    """A function to create duplicate images form the training dataset for the image_augmentation.
          Parameters
          ----------
          x : x is the image dataset.
          y : y is the class label.

          Steps
          ----------
          STEP 1 : Finds the unique class labels.
          STEP 2 : Fetches the 100 images per class
          STEP 3 : Appends the populate images and its labels to a list.

          Returns
          ----------
          images : 100 images per class is returned.
          labels : The lables of the images are returned.
          """

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


