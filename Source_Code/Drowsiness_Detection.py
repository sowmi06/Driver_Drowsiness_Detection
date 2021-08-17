from face_eye_detector import face_detector
from face_eye_detector import eye_detector
from face_eye_detector import label_encoder
from face_eye_detector import split
from face_eye_detector import populate_images
from tensorflow.keras.applications import VGG16, ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import time
from sklearn.decomposition import PCA
import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt


def image_augmentation(x_train, y_train):
    """A function for augmenting the newly populated images.
           Parameters
           ----------
           x_train : The x_train images of ndarray.
           y_train : The y_train class labels of ndarray.

           Steps
           ----------
           STEP 1 : Populates 100 images per class using the created populate_images() function.
           STEP 2 : Uses an in-built ImageDataGenerator() method to do a flip and shift the width and height of images.
           STEP 3 : Obtains the newly augmented images.
           STEP 4 : Combine the newly augmented images and its labels to the untouched x_train images.

           Returns
           ----------
           nd-array:
                x_train_augmented with combined augmented and original x_train images.
                y_train_augmented with the x_train_augmented labels.
           """

    images, labels = populate_images(x_train, y_train)
    data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)
    train_generator = data_generator.flow(images, labels, batch_size=500)
    aug_images = train_generator[0][0]
    aug_labels = train_generator[0][1]
    x_train_augmented = np.vstack((x_train, aug_images))
    y_train_augmented = np.hstack((y_train, aug_labels))

    return x_train_augmented, y_train_augmented


def VGG16_model(x_train_augmented, x_test):
    """A function extract deep features from the VGG16 model.
           Parameters
           ----------
           x_train_augmented : The x_train images of ndarray to extract deep features.
           x_test : The x_train images of ndarray to extract deep features.

           Steps
           ----------
           STEP 1 : Downloads the VGG16 CNN model with input shape=(224, 224, 3), imagenet weights
                    and without the fully connected layer
           STEP 2 : Use the GlobalAveragePooling2D() function to summarize the important features maps.
           STEP 3 : Construct the model of VGG16.
           STEP 4 : Input the x_train_augmented and x_test to extract deep features

           Returns
           ----------
           nd-array:
                vgg_train_deep_features : Extracted train deep features from the pre-trained imagenet
                                          CNN model.
                vgg_test_deep_features : Extracted test deep features from the pre-trained imagenet
                                         CNN model.
           """

    base = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    out = base.output
    out = GlobalAveragePooling2D()(out)
    out = Flatten()(out)
    vgg = Model(inputs=base.input, outputs=out)
    # extracting deep features on the x_train and x_test
    vgg_train_deep_features = vgg.predict(x_train_augmented)
    vgg_test_deep_features = vgg.predict(x_test)

    return vgg_train_deep_features, vgg_test_deep_features


def ResNet101_model(x_train_augmented, x_test):
    """A function extract deep features from the RESNET101 model.
           Parameters
           ----------
           x_train_augmented : The x_train images of ndarray to extract deep features.
           x_test : The x_train images of ndarray to extract deep features.

           Steps
           ----------
           STEP 1 : Downloads the ResNet101 model with input shape=(224, 224, 3), imagenet weights
                    and without the fully connected layer.
           STEP 2 : Use the GlobalAveragePooling2D() function to summarize the important features maps.
           STEP 3 : Construct the model of ResNet101.
           STEP 4 : Input the x_train_augmented and x_test to extract deep features.

           Returns
           ----------
           nd-array:
                resnet_train_deep_features : Extracted train deep features from the pre-trained imagenet
                                             CNN model.
                resnet_test_deep_features : Extracted test deep features from the pre-trained imagenet
                                            CNN model.
           """

    base = ResNet101(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    out = base.output
    out = GlobalAveragePooling2D()(out)
    out = Flatten()(out)
    resnet_model = Model(inputs=base.input, outputs=out)
    # extracting deep features on the x_train and x_test
    resnet_train_deep_features = resnet_model.predict(x_train_augmented)
    resnet_test_deep_features = resnet_model.predict(x_test)

    return resnet_train_deep_features, resnet_test_deep_features


def feature_combination(vgg_train_deep_features, vgg_test_deep_features, resnet_train_deep_features, resnet_test_deep_features):
    """A function to combine the extracted deep features from the vgg16 and resnet101.
           Parameters
           ----------
           vgg_train_deep_features : Extracted train deep features from the pre-trained imagenet CNN model.
           vgg_test_deep_features : Extracted test deep features from the pre-trained imagenet CNN model.
           resnet_train_deep_features : Extracted train deep features from the pre-trained imagenet CNN model.
           resnet_test_deep_features : Extracted test deep features from the pre-trained imagenet CNN model.

           Steps
           ----------
           STEP 1 : Combines the vgg and resnet train features together horizontally(column-wise).
           STEP 2 : Combines the vgg and resnet test features together horizontally(column-wise).

           Returns
           ----------
           nd-array:
                x_train_deep_features : Combined train deep feature from the vgg16 and resnet101.
                x_test_deep_features : Combined test deep feature from the vgg16 and resnet101.
           """

    # combining the train deep features
    x_train_deep_features = np.hstack((vgg_train_deep_features, resnet_train_deep_features))
    # combining the test deep features
    x_test_deep_features = np.hstack((vgg_test_deep_features, resnet_test_deep_features))

    return x_train_deep_features, x_test_deep_features


def pricipal_component_analysis(x_train_deep_features, x_test_deep_features, y_train_augmented):
    """A function to use Principal component analysis (Dimentionality reduction) method.
          Parameters
          ----------
          x_train_deep_features : Combined train deep feature from the vgg16 and resnet101.
          x_test_deep_features : Combined test deep feature from the vgg16 and resnet101.

          Steps
          ----------
          STEP 1 : Calling an object using an in-built PCA method.
          STEP 2 : fit_transfom() the x_train_deep_features and transform() x_test_deep_features to reduce
                   the dimension of the features.

          Returns
          ----------
          nd-array:
               x_train_deep_features : Dimension reduced train deep features.
               x_test_deep_features : Dimension reduced test deep features.
          """

    pca = PCA()
    x_train_deep_features = pca.fit_transform(x_train_deep_features, y_train_augmented)
    x_test_deep_features = pca.transform(x_test_deep_features)

    return x_train_deep_features, x_test_deep_features


def Svm_Classifier(x_train_deep_features, x_test_deep_features, y_train_augmented, y_test):
    """A function to classify and evaluate using SVM Classifier .
          Parameters
          ----------
          x_train_deep_features : Combined and Dimension reduced train deep features.
          x_test_deep_features : Combined and Dimension reduced test deep features.

          Steps
          ----------
          STEP 1 : Calling an in-built SVM classifier form sk-learn with "rbf" kernel and C=0.9.
          STEP 2 : Train the classifier using fit() method with x_train_deep_features and y_train_augmented values.
          STEP 3 : Record the training time to print
          STEP 4 : Predict the classifier using predict() method with x_train_deep_features and y_train_augmented values
                   which will output the predicted y test values.
          STEP 5 : Evaluate and print the SVM model using accuracy, precision, recall and confusion matrix values.

          Returns
          ----------
          nd-array:
               x_train_deep_features : Dimension reduced train deep features.
               x_test_deep_features : Dimension reduced test deep features.
          """

    # initializing the svm classifier
    classifier = SVC(kernel='rbf', C=0.9)

    # start train time
    start_time = time.process_time()

    # training the classifier
    print("-------------------------------")
    print(" Training using SVM ")
    classifier.fit(x_train_deep_features, y_train_augmented)
    y_train_predicted = classifier.predict(x_train_deep_features)

    # end train time
    end_time = time.process_time()

    # calculate train time
    training_time = end_time - start_time

    # testing the classifier
    y_test_predicted = classifier.predict(x_test_deep_features)
    print("-------------------------------")
    print(" Testing using SVM")
    print("-------------------------------")

    # Evaluate and print the SVM model using accuracy, precision, recall and confusion matrix
    training_accuracy = accuracy_score(y_train_augmented, y_train_predicted)
    testing_accuracy = accuracy_score(y_test, y_test_predicted)
    precision = precision_score(y_test, y_test_predicted, average='macro')
    recall = recall_score(y_test, y_test_predicted, average='macro')
    con_mat = confusion_matrix(y_test, y_test_predicted)

    print("Training time :", training_time)
    print("Training Accuracy : {}".format(training_accuracy))
    print("Testing Accuracy : {}".format(testing_accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("confusion matrix", con_mat)
    print("-------------------------------")


def main():
    # specifies the face path and face_cascade_path
    face_path = "../Dataset/Face"
    face_cascade_path = "../Cascade_Classifier/haarcascade_frontalface_default.xml"

    # specifies the eye path and eye_cascade_path
    eye_path = "../Dataset/Eyes"
    eye_cascade_path = "../Cascade_Classifier/haarcascade_eye.xml"

    # Extract the Face and Eye images from the dataset
    face_image, face_label = face_detector(face_path, face_cascade_path)
    eye_image, eye_label = eye_detector(eye_path, eye_cascade_path)

    # Data Preprocessing
    X = np.concatenate((face_image, eye_image), axis=0)
    y = np.concatenate((face_label, eye_label), axis=0)

    # Processing label encoding for class labels
    Y = label_encoder(y)

    # Making 70:30 train-test split
    x_train, x_test, y_train, y_test = split(X, Y)

    # Performing Image data argumentation on the train dataset
    x_train_augmented, y_train_augmented = image_augmentation(x_train, y_train)

    # Extracting deep features using pre-trained VGG-16 with imagenet weights
    vgg_train_deep_features, vgg_test_deep_features = VGG16_model(x_train_augmented, x_test)

    # Extracting deep features using pre-trained ResNet101 with imagenet  weights
    resnet_train_deep_features, resnet_test_deep_features = ResNet101_model(x_train_augmented, x_test)

    # Combination VGG16 and Resnet101 deep features
    x_train_deep_features, x_test_deep_features = feature_combination(vgg_train_deep_features, vgg_test_deep_features,
                                                                      resnet_train_deep_features,resnet_test_deep_features)

    # Using PCA for dimentionality reduction
    x_train_deep_features, x_test_deep_features = pricipal_component_analysis(x_train_deep_features, x_test_deep_features, y_train_augmented)

    # predicting and evaluating metrics using SVM classifier
    Svm_Classifier(x_train_deep_features, x_test_deep_features, y_train_augmented, y_test)


if __name__ == "__main__":
    main()