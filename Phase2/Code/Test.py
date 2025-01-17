#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import argparse
import math as m
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import torch
from Network.Network import Basic_CNN
from Network.Modified_network import Advanced_CNN
from Network.Resnet_manual import ResnetBasic
from Network.Resnext_manual import ResnextBasic
from Network.DenseNet import densenet_small
from Misc.MiscUtils import *
from Misc.DataUtils import *
import torchvision.transforms as transforms
from functions.fancy_plots import *
from torchviz import make_dot
import netron
import seaborn as sns
import time


# Don't generate pyc codes
sys.dont_write_bytecode = True


def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    Img = Img * 2 - 1.0

    ##########################################################################
    return Img


def ReadData(TrainSet, TrainLabels, ImageSize, Images_path, k):
    Imagename = TrainSet + "/" + Images_path[k] + ".png"

    image_bgr = cv2.imread(Imagename)

    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at {Imagename}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),  # Randomly flip the image horizontally with a probability of 0.5
            transforms.RandomVerticalFlip(p=0.5),
        ]  # Converts to tensor and normalizes to [0, 1]
    )
    image_tensor = transform(image_rgb)

    I1 = image_tensor
    I1 = I1 * 2 - 1.0
    I1 = I1.unsqueeze(0)
    Label = int(TrainLabels[k])
    Label = np.array([Label])
    Label = torch.tensor(Label)

    return I1, Label


def ReadImages(Img):
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1


def Accuracy(Pred, GT):
    return np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred)


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if not (os.path.isfile(LabelsPathTest)):
        print("ERROR: Test Labels do not exist in " + LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, "r")
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if not (os.path.isfile(LabelsPathPred)):
        print("ERROR: Pred Labels do not exist in " + LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, "r")
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelTest, LabelPred


def ConfusionMatrix(LabelsTrue, LabelsPred, PlotPath, class_names, name):

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(
        y_true=LabelsTrue, y_pred=LabelsPred  # True class for test-set.
    )  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + " ({0})".format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print("Accuracy: " + str(Accuracy(LabelsPred, LabelsTrue)), "%")

    # Save accuracy to a text file
    accuracy_file_path = f"{PlotPath}/{name}_accuracy.txt"
    with open(accuracy_file_path, "w") as file:
        file.write(f"Accuracy: {Accuracy(LabelsPred, LabelsTrue):.2f} %\n")

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 7))

    # Use Seaborn to create a heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    # Labels, title and ticks
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    # Save the plot
    plt.savefig(f"{PlotPath}/{name}.pdf", bbox_inches="tight")
    plt.close()

    return None


def TestOperation(
    ImageSize,
    ModelPath,
    TestSet,
    LabelsPathPred,
    Epochs,
    model_type,
    labels_data,
    images_path,
):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    print("----------------------------------------------------")
    print(images_path[0:3])
    print("----------------------------------------------------")
    # Folder to save results
    name_results = "Results"
    path = os.path.join(os.getcwd(), name_results)
    os.makedirs(path, exist_ok=True)

    # Get GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Neural network model
    simple_model = Basic_CNN(InputSize=3, OutputSize=10).to(device)
    advanced_model = Advanced_CNN(InputSize=(3, 32, 32), OutputSize=10).to(device)
    resnet_model = ResnetBasic(InputSize=(3, 32, 32), OutputSize=10).to(device)
    resnext_model = ResnextBasic(InputSize=(3, 32, 32), OutputSize=10).to(device)
    densenet = densenet_small().to(device)

    # Check for the structure
    if model_type == "simple":
        model = simple_model
    elif model_type == "advanced":
        model = advanced_model
    elif model_type == "resnet":
        model = resnet_model
    elif model_type == "resnext":
        model = resnext_model
    elif model_type == "dense":
        model = densenet
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Acess a folder with the NN weights
    CheckPointPath = os.path.join(ModelPath, model.name)
    os.makedirs(CheckPointPath, exist_ok=True)

    # Create Folder with the name of the strcuture
    path_results_network = os.path.join(path, model.name)
    os.makedirs(path_results_network, exist_ok=True)

    ## Init Empty Data
    loss_epochs = []
    accuracy_epochs = []
    epochs = []

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Aux variable
    labels_data = list(labels_data)
    print("--------------------------------------")
    print(model.name)

    # Learning over epochs
    for k in range(0, Epochs):
        ModelPath_aux = CheckPointPath + "/" + str(k) + "model.ckpt"
        CheckPoint = torch.load(ModelPath_aux)
        model.load_state_dict(CheckPoint["model_state_dict"])
        OutSaveT = open(LabelsPathPred, "w")
        loss_per_iteration = []
        accuracy_per_iteration = []
        model.eval()

        # Moving along the data
        for count in tqdm(range(len(images_path))):
            images, labels = ReadData(
                TestSet, labels_data, ImageSize, images_path, count
            )

            images, labels = images.to(device), labels.to(device)

            # Predictions
            PredT = torch.argmax(model(images)).item()

            # Save results
            result = model.validation_step((images, labels))
            loss_per_iteration.append(result["loss"].item())
            accuracy_per_iteration.append(result["acc"].item())

            OutSaveT.write(str(PredT) + "\n")
        OutSaveT.close()
        # Save loss

        loss_epochs.append(np.mean(loss_per_iteration))
        accuracy_epochs.append(np.mean(accuracy_per_iteration))
        epochs.append(k + 1)

        print(f"Cost per epochs: {np.mean(loss_per_iteration)}")

    # Create numpy data
    loss_epochs = np.array(loss_epochs)
    accuracy_epochs = np.array(accuracy_epochs)
    epochs = np.array(epochs)
    data_test = np.vstack((epochs, loss_epochs, accuracy_epochs))

    # Save Data as numpy
    name = os.path.join(path_results_network, f"Data_test.npy")
    np.save(name, data_test)

    # Names for the Neural network structure
    name_network = os.path.join(path_results_network, f"NN")
    name_onnix = name_network + "." + "onnx"

    # Load Trainning Data
    name_tranning = os.path.join(path_results_network, f"Data_trainning.npy")
    load_tranning_values = np.load(name_tranning)

    # Get Values for the plots
    data_loss = np.vstack((data_test[1, :], load_tranning_values[1, :]))
    data_accuracy = np.vstack((data_test[2, :], load_tranning_values[2, :]))

    # Plot Data
    fig11, ax11, ax12 = fancy_plots_2()
    plot_states_full(
        fig11,
        ax11,
        ax12,
        data_loss,
        data_accuracy,
        data_test[0, :],
        "Results",
        path_results_network,
    )

    # Save Image Neural Network
    make_dot(model(images), params=dict(model.named_parameters())).render(
        name_network, format="png"
    )

    # Export Neural Network just to visualize the structure
    torch.onnx.export(
        model,
        images,
        name_onnix,
        input_names=["Image"],
        output_names=["Probabilities"],
        opset_version=11,
    )

    print("----------------------------------------------------")
    print(images_path[0:3])
    print("----------------------------------------------------")
    return name_onnix, path_results_network


def TrainningOperation(
    ImageSize, ModelPath, TestSet, LabelsPathPred, model_type, labels_data, images_path
):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    print("----------------------------------------------------")
    print(images_path[0:3])
    print("----------------------------------------------------")
    # Check for results Folder
    name_results = "Results"
    path = os.path.join(os.getcwd(), name_results)
    os.makedirs(path, exist_ok=True)

    # Check for Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init all the structures
    simple_model = Basic_CNN(InputSize=3, OutputSize=10).to(device)
    advanced_model = Advanced_CNN(InputSize=(3, 32, 32), OutputSize=10).to(device)
    resnet_model = ResnetBasic(InputSize=(3, 32, 32), OutputSize=10).to(device)
    resnext_model = ResnextBasic(InputSize=(3, 32, 32), OutputSize=10).to(device)
    densenet = densenet_small().to(device)

    # Check for the structure
    if model_type == "simple":
        model = simple_model
    elif model_type == "advanced":
        model = advanced_model
    elif model_type == "resnet":
        model = resnet_model
    elif model_type == "resnext":
        model = resnext_model
    elif model_type == "dense":
        model = densenet
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    # Acess a folder with the NN weights
    CheckPointPath = os.path.join(ModelPath, model.name)
    os.makedirs(CheckPointPath, exist_ok=True)

    # Create a sublder where we can save the results
    path_results_network = os.path.join(path, model.name)
    os.makedirs(path_results_network, exist_ok=True)

    # Load the last values
    ModelPath_aux = CheckPointPath + "/" + "29" + "model.ckpt"
    print(ModelPath_aux)
    CheckPoint = torch.load(ModelPath_aux)
    model.load_state_dict(CheckPoint["model_state_dict"])
    OutSaveT = open(LabelsPathPred, "w")

    ## Save parameters
    name_parameters = path_results_network + "/" + "model_parameters.txt"
    with open(name_parameters, "w") as f:
        f.write("Model's state_dict:\n")
        for param_tensor in model.state_dict():
            f.write(f"{param_tensor}\t {model.state_dict()[param_tensor].size()}\n")

    model.eval()

    print("--------------------------------------")
    print(model.name)
    # Aux variable
    labels_data = list(labels_data)
    for count in tqdm(range(len(images_path))):
        images, labels = ReadData(TestSet, labels_data, ImageSize, images_path, count)

        images, labels = images.to(device), labels.to(device)

        # Predictions
        PredT = torch.argmax(model(images)).item()

        OutSaveT.write(str(PredT) + "\n")
    OutSaveT.close()

    print("----------------------------------------------------")
    print(images_path[0:3])
    print("----------------------------------------------------")
    return None


def CheckSpeed(ImageSize, ModelPath, TestSet, model_type, labels_data, images_path):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    print("----------------------------------------------------")
    print("Check Velocity")
    print("----------------------------------------------------")

    # Check for Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init all the structures
    simple_model = Basic_CNN(InputSize=3, OutputSize=10).to(device)
    advanced_model = Advanced_CNN(InputSize=(3, 32, 32), OutputSize=10).to(device)
    resnet_model = ResnetBasic(InputSize=(3, 32, 32), OutputSize=10).to(device)
    resnext_model = ResnextBasic(InputSize=(3, 32, 32), OutputSize=10).to(device)
    densenet = densenet_small().to(device)
    model = [simple_model, advanced_model, resnet_model]

    print("--------------------------------------")
    # Aux variable
    labels_data = list(labels_data)

    # Variables to save data
    number_of_neutwoks = 3
    # NUmber of images
    image_number = 1000

    # Time matrix
    sampling_time = np.zeros((number_of_neutwoks, image_number))

    for k in range(0, number_of_neutwoks):
        CheckPointPath = os.path.join(ModelPath, model[k].name)
        os.makedirs(CheckPointPath, exist_ok=True)

        # Load the last values
        ModelPath_aux = CheckPointPath + "/" + "29" + "model.ckpt"
        print(ModelPath_aux)

        CheckPoint = torch.load(ModelPath_aux)
        model[k].load_state_dict(CheckPoint["model_state_dict"])

        model[k].eval()
        for count in tqdm(range(image_number)):
            tic = time.time()
            images, labels = ReadData(
                TestSet, labels_data, ImageSize, images_path, count
            )
            images, labels = images.to(device), labels.to(device)

            # Predictions
            PredT = torch.argmax(model[k](images)).item()
            toc = time.time() - tic
            sampling_time[k, count] = toc

    average_values = np.mean(sampling_time, axis=1)

    model_2 = [resnext_model, densenet]

    print("--------------------------------------")
    # Variables to save data
    number_of_neutwoks = 2
    # NUmber of images
    image_number = 1000

    # Time matrix
    sampling_time_2 = np.zeros((number_of_neutwoks, image_number))

    for k in range(0, number_of_neutwoks):
        CheckPointPath = os.path.join(ModelPath, model_2[k].name)
        os.makedirs(CheckPointPath, exist_ok=True)

        # Load the last values
        ModelPath_aux = CheckPointPath + "/" + "49" + "model.ckpt"
        print(ModelPath_aux)

        CheckPoint = torch.load(ModelPath_aux)
        model_2[k].load_state_dict(CheckPoint["model_state_dict"])

        model_2[k].eval()
        for count in tqdm(range(image_number)):
            tic = time.time()
            images, labels = ReadData(
                TestSet, labels_data, ImageSize, images_path, count
            )
            images, labels = images.to(device), labels.to(device)

            # Predictions
            PredT = torch.argmax(model_2[k](images)).item()
            toc = time.time() - tic
            sampling_time_2[k, count] = toc

    average_values_2 = np.mean(sampling_time_2, axis=1)

    print("-----------Time to for the Inference---------------------------")
    print(average_values)
    print(average_values_2)

    print("----------------------------------------------------")
    print(images_path[0:3])
    print("----------------------------------------------------")
    return None


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="../Checkpoints",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )

    Parser.add_argument(
        "--LabelsPathTrainning",
        dest="LabelsPathTrainning",
        default="./TxtFiles/LabelsTrain.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTrain.txt",
    )

    Parser.add_argument(
        "--NumEpochs",
        dest="NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs that we set up for the trainning",
    )
    Parser.add_argument(
        "--type",
        default="simple",
        help="Type of neural network",
    )

    Parser.add_argument(
        "--ImagePath",
        default="../CIFAR10",
        help="Base path of images",
    )

    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    # Variables of the system
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsPath = Args.LabelsPath
    LabelsPathTrainning = Args.LabelsPathTrainning
    Epochs = Args.NumEpochs
    type_NN = Args.type
    Images_path = Args.ImagePath
    CheckPointPath = Args.ImagePath

    # Get Directories for the test data
    (
        Dirname_test,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TestLabels,
        NumClasses,
    ) = SetupAllTest(Images_path, CheckPointPath)

    # Get Directories for the training data
    (
        Dirname_training,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainingLabels,
        NumClasses,
    ) = SetupAll(Images_path, CheckPointPath)

    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = "./TxtFiles/PredOut.txt"
    LabelsPathPred_trainning = "./TxtFiles/PredOut_trainning.txt"

    name_onnix, path_results = TestOperation(
        ImageSize,
        ModelPath,
        Images_path,
        LabelsPathPred,
        Epochs,
        type_NN,
        TestLabels,
        Dirname_test,
    )
    # Trainning results
    TrainningOperation(
        ImageSize,
        ModelPath,
        Images_path,
        LabelsPathPred_trainning,
        type_NN,
        TrainingLabels,
        Dirname_training,
    )

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    LabelsTrue_tranning, LabelsPred_tranning = ReadLabels(
        LabelsPathTrainning, LabelsPathPred_trainning
    )

    # Show Confusion matrix
    ConfusionMatrix(LabelsTrue, LabelsPred, path_results, None, "test")
    ConfusionMatrix(
        LabelsTrue_tranning, LabelsPred_tranning, path_results, None, "trainning"
    )

    # Show neural network arquitecture
    netron.start(name_onnix)


if __name__ == "__main__":
    main()
