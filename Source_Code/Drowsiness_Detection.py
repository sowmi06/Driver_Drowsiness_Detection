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
from sklearn import svm
import numpy as np
import time
from sklearn.decomposition import PCA
import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt


def image_augmentation(x_train, y_train):
    images, labels = populate_images(x_train, y_train)
    data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)
    train_generator = data_generator.flow(images, labels, batch_size=500)
    aug_images = train_generator[0][0]
    aug_labels = train_generator[0][1]
    augmented_x_train = np.vstack((x_train, aug_images))
    augmented_y_train = np.hstack((y_train, aug_labels))
    return augmented_x_train, augmented_y_train


def VGG_model(augmented_x_train, x_test):
    base = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    out = base.output
    out = GlobalAveragePooling2D()(out)
    out = Flatten()(out)
    model = Model(inputs=base.input, outputs=out)
    # extracting deep features on the x_train and x_test
    vgg_train_deep_features = model.predict(augmented_x_train)
    vgg_test_deep_features = model.predict(x_test)

    return vgg_train_deep_features, vgg_test_deep_features


def ResNet_model(augmented_x_train, x_test):
    base = ResNet101(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    out = base.output
    out = GlobalAveragePooling2D()(out)
    out = Flatten()(out)
    model = Model(inputs=base.input, outputs=out)
    # extracting deep features on the x_train and x_test
    resnet_train_deep_features = model.predict(augmented_x_train)
    resnet_test_deep_features = model.predict(x_test)

    return resnet_train_deep_features, resnet_test_deep_features


def feature_combination(vgg_train_deep_features, vgg_test_deep_features, resnet_train_deep_features, resnet_test_deep_features):
    # combining the train deep features
    x_train_deep_features = np.hstack((vgg_train_deep_features, resnet_train_deep_features))
    # combining the test deep features
    x_test_deep_features = np.hstack((vgg_test_deep_features, resnet_test_deep_features))

    return x_train_deep_features, x_test_deep_features


def pricipal_component_analysis(x_train_deep_features, x_test_deep_features, train_y):
    pca = PCA()
    x_train_deep_features = pca.fit_transform(x_train_deep_features, train_y)
    x_test_deep_features = pca.transform(x_test_deep_features)

    return x_train_deep_features, x_test_deep_features


def Svm_Classifier(x_train_deep_features, x_test_deep_features, train_y, y_test):

    # initializing the svm classifier
    classifier = svm.SVC(kernel='rbf', C=0.9)

    # training the classifier
    start_time = time.process_time()

    classifier.fit(x_train_deep_features, train_y)
    y_train_predicted = classifier.predict(x_train_deep_features)

    end_time = time.process_time()

    training_time = end_time - start_time

    # testing the classifier
    y_predicted = classifier.predict(x_test_deep_features)


    """" Evaluating SVM model """""
    # Evaluate the SVM model using accuracy, precision, recall and confusion matrix
    training_accuracy = accuracy_score(train_y, y_train_predicted)
    testing_accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='macro')
    recall = recall_score(y_test, y_predicted, average='macro')
    con_mat = confusion_matrix(y_test, y_predicted)

    print("Training time :", training_time)
    print("Training Accuracy : {}".format(training_accuracy))
    print("Testing Accuracy : {}".format(testing_accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    # print("F1-Score : {}".format(f1score))
    print("confusion matrix", con_mat)


def data_visualization():
    pass



def main():
    face_path = "../Dataset/Face"
    face_cascade_path = "../Cascade_Classifier/haarcascade_frontalface_default.xml"

    eye_path = "../Dataset/Eyes"
    eye_cascade_path = "../Cascade_Classifier/haarcascade_eye.xml"

    # Extract the Face and Eye images from the dataset
    face_image, face_label = face_detector(face_path, face_cascade_path)
    eye_image, eye_label = eye_detector(eye_path, eye_cascade_path)

    # Data Preprocessing
    X = np.concatenate((face_image, eye_image), axis=0)
    y = np.concatenate((face_label, eye_label), axis=0)
    Y = label_encoder(y)

    # Making train test split
    x_train, x_test, y_train, y_test = split(X, Y)

    # image data argumentation on the train dataset
    augmented_x_train, augmented_y_train = image_augmentation(x_train, y_train)

    # extracting deep features using VGG-16 places365 weights
    vgg_train_deep_features, vgg_test_deep_features = VGG_model(augmented_x_train, x_test)

    # extracting deep features using ResNet101  weights
    resnet_train_deep_features, resnet_test_deep_features = ResNet_model(augmented_x_train, x_test)

    # feature combination
    x_train_deep_features, x_test_deep_features = feature_combination(vgg_train_deep_features, vgg_test_deep_features,
                                                                      resnet_train_deep_features,resnet_test_deep_features)

    x_train_deep_features, x_test_deep_features = pricipal_component_analysis(x_train_deep_features, x_test_deep_features, augmented_y_train)


    # predicting using SVM classifier
    Svm_Classifier(x_train_deep_features, x_test_deep_features, augmented_y_train, y_test)

    temp=0


if __name__ == "__main__":
    main()