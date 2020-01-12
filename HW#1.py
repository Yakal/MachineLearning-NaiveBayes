__author__ = "Furkan Yakal"
__email__ = "fyakal16@ku.edu.tr"

import numpy as np
import pandas as pd
import time

start = time.time()
# Read the images and labels
images = pd.read_csv("hw01_images.csv", header=None)
labels = pd.read_csv("hw01_labels.csv", header=None)

# 1 is Female, 2 is Male
classnum_to_gender = {0: "F", 1: "M"}
gender_to_classnum = {"F": 0, "M": 1}

# training set is formed from first 200 images
train_set_images = np.array(images[:200])
train_set_labels = np.array(labels[:200])

# test set is formed from remaining 200 images
test_set_images = np.array(images[200:])
test_set_labels = np.array(labels[200:])

# number of female and male labels in training set
number_of_female_label = len(train_set_labels[train_set_labels == 1])
number_of_male_label = len(train_set_labels[train_set_labels == 2])

# mean estimations of female and male
mean_estimation_female = np.sum(train_set_images[np.reshape(train_set_labels == 1, 200)],
                                axis=0) / number_of_female_label
mean_estimation_male = np.sum(train_set_images[np.reshape(train_set_labels == 2, 200)], axis=0) / number_of_male_label

# standart deviation estimations of female and male
std_estimation_female = np.sqrt(
    np.sum((train_set_images[np.reshape(train_set_labels == 1, 200)] - mean_estimation_female) ** 2,
           axis=0) / number_of_female_label)
std_estimation_male = np.sqrt(
    np.sum((train_set_images[np.reshape(train_set_labels == 2, 200)] - mean_estimation_male) ** 2,
           axis=0) / number_of_male_label)

# prior probabilities of female and male
prior_prob_female = number_of_female_label / len(train_set_labels)
prior_prob_male = number_of_male_label / len(train_set_labels)


# calculate the score values with prior probabilities
def score_function(st_dev, s_mean, prior, image):
    return np.log(prior) + np.sum(
        (np.log((st_dev ** 2) * np.pi * 2) * -0.5) + ((((image - s_mean) ** 2) / (st_dev ** 2)) * -0.5))


# returns the class with highest score
def predict_class(image):
    whole_set = [score_function(std_estimation_female, mean_estimation_female, prior_prob_female, image),
                 score_function(std_estimation_male, mean_estimation_male, prior_prob_male, image)]
    loc = int(np.argmax(whole_set))
    return classnum_to_gender[loc]


# generates the confusion matrix
def confusion_matrix(image_set, label_set):
    y_true = label_set
    y_prediction = np.apply_along_axis(func1d=predict_class, arr=image_set, axis=1)
    conf_matrix = np.zeros([2, 2], dtype=np.int32)
    for i in range(len(y_true)):
        conf_matrix[gender_to_classnum[y_prediction[i]], y_true[i][0]-1] += 1
    return conf_matrix


# prints the matrix in the desired format
def print_conf_matrix(data_set_type, matrix):
    print("\n\n----------------{}----------------".format(data_set_type))
    print("\t\t\t\t y_correct \t \t")
    print("y_estimation \t Female \t Male")
    print("Female \t\t\t {} \t\t {}".format(matrix[0, 0], matrix[0, 1]))
    print("Male \t\t\t {} \t\t\t {}".format(matrix[1, 0], matrix[1, 1]))


def print_calculated_results():
    print("--------------------------------------------------------------\n")
    print("shape of training_set_images is as follows: {}\n".format(np.shape(train_set_images)))
    print("shape of training_set_labels is as follows: {}\n".format(np.shape(train_set_labels)))
    print("shape of test_set_images is as follows: {}\n".format(np.shape(test_set_images)))
    print("shape of test_set_labels is as follows: {}\n".format(np.shape(test_set_labels)))
    print("--------------------------------------------------------------\n")
    print("mean estimation of female:\n{}\n".format(mean_estimation_female))
    print("mean estimation of male:\n{}\n".format(mean_estimation_male))
    print("--------------------------------------------------------------\n")
    print("std estimation of female:\n{}\n".format(std_estimation_female))
    print("std estimation of male:\n{}\n".format(std_estimation_male))
    print("--------------------------------------------------------------\n")
    print("prior probability of female {}\n".format(prior_prob_female))
    print("prior probability of male {}\n".format(prior_prob_male))
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    print_calculated_results()
    # TRAIN MATRIX
    train_matrix = confusion_matrix(train_set_images, train_set_labels)
    print_conf_matrix("TRAIN", train_matrix)

    # TEST MATRIX
    test_matrix = confusion_matrix(test_set_images, test_set_labels)
    print_conf_matrix("TEST", test_matrix)
    finish = time.time()
    print("\nIt takes {} seconds".format(finish-start))

