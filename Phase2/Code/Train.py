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


import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
import sys
import os
import numpy as np
import random
import os
import random
import matplotlib.pyplot as plt
import argparse
import math as m
from tqdm.notebook import tqdm
from Network.Network import Basic_CNN, accuracy
from Network.Modified_network import Advanced_CNN
from Network.Resnet_manual import ResnetBasic
from Network.Resnext_manual import ResnextBasic
from Network.DenseNet import densenet_small
from Misc.MiscUtils import *
from Misc.DataUtils import *
import torch.optim as optim
import torch.nn.functional as F
from functions.fancy_plots import *
from torchsummary import summary
import torchvision.transforms as transforms
import netron


# Don't generate pyc codes
sys.dont_write_bytecode = True


def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize, Images_path):
    """
    Inputs:
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch

    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels
    """
    I1Batch = []
    LabelBatch = []
    ImageNum = 0

    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(Images_path) - 1)
        ImageNum += 1
        Imagename = TrainSet + "/" + Images_path[RandIdx] + ".png"

        ############################################################
        ### Add any standardization or data augmentation here!
        ############################################################
        image_bgr = cv2.imread(Imagename)

        if image_bgr is None:
            raise FileNotFoundError(f"Image not found at {Imagename}")

        ### Convert the image to RGB format
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

        # I1, Label = TrainSet[RandIdx]

        I1 = I1 * 2 - 1.0

        Label = int(TrainLabels[RandIdx])

        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))

    return torch.stack(I1Batch), torch.stack(LabelBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    TrainLabels,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    TrainSet,
    LogsPath,
    model_type,
    directory_train,
):
    """
    Inputs:
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """

    print("----------------------------------------------------")
    print(directory_train[0:3])
    print("----------------------------------------------------")

    # Define the location of the folder, where we can save the results
    name_results = "Results"
    path = os.path.join(os.getcwd(), name_results)
    os.makedirs(path, exist_ok=True)

    # Ceck for torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # neural Networks definitions
    simple_model = Basic_CNN(InputSize=3, OutputSize=10).to(device)
    advanced_model = Advanced_CNN(InputSize=(3, 32, 32), OutputSize=10).to(device)
    resnet_model = ResnetBasic(InputSize=(3, 32, 32), OutputSize=10).to(device)
    resnext_model = ResnextBasic(InputSize=(3, 32, 32), OutputSize=10).to(device)
    densenet = densenet_small().to(device)

    # Switch between them
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

    # Create a folder with the name of the structure
    path_results_network = os.path.join(path, model.name)
    os.makedirs(path_results_network, exist_ok=True)

    # Define the optimizer for eah neural network
    Optimizer_basic = optim.Adam(model.parameters(), lr=0.001)
    Optimizer_advance = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    Optimizer_resnet = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    Optimizer_resnext = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    Optimizer_densenet = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    if model_type == "simple":
        Optimizer = Optimizer_basic
    elif model_type == "advanced":
        Optimizer = Optimizer_advance
    elif model_type == "resnet":
        Optimizer = Optimizer_resnet
    elif model_type == "resnext":
        Optimizer = Optimizer_resnext
    elif model_type == "dense":
        Optimizer = Optimizer_densenet
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Learning decay
    decay = optim.lr_scheduler.StepLR(Optimizer, step_size=20, gamma=0.01)

    # Tensor board section
    Writer = SummaryWriter(LogsPath)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    summary(model, input_size=(3, 32, 32))

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    # Aux variables to the safe the cost along the epochs
    loss_epochs = []
    accuracy_epochs = []
    epochs = []

    # Create directory for the neural network and its weights
    CheckPointPath = os.path.join(CheckPointPath, model.name)
    os.makedirs(CheckPointPath, exist_ok=True)

    # Aux variable, possible bug
    TrainLabels = list(TrainLabels)

    print(model.name)

    # Learning over epochs
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        loss_per_iteration = []
        accuracy_per_iteration = []
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)

        # Learning over the bachs
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            images, labels = GenerateBatch(
                TrainSet, TrainLabels, ImageSize, MiniBatchSize, directory_train
            )
            images, labels = images.to(device), labels.to(device)

            ## Predict output with forward pass
            LossThisBatch = model.training_step((images, labels))

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + "/"
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
            result = model.validation_step((images, labels))
            loss_per_iteration.append(result["loss"].item())
            accuracy_per_iteration.append(result["acc"].item())

            Writer.add_scalar(
                "LossEveryIter",
                result["loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            Writer.add_scalar(
                "Accuracy",
                result["acc"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + "/" + str(Epochs) + "model.ckpt"

        # Save loss
        loss_epochs.append(np.mean(loss_per_iteration))
        accuracy_epochs.append(np.mean(accuracy_per_iteration))
        epochs.append(Epochs + 1)
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")

        if model_type == "simple":
            pass

        elif model_type == "advanced":
            decay.step()

        elif model_type == "resnet":
            decay.step()

        elif model_type == "resnext":
            decay.step()

        elif model_type == "dense":
            decay.step()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        current_lr = Optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr}")
        print(f"Cost per epochs: {np.mean(loss_per_iteration)}")

    loss_epochs = np.array(loss_epochs)
    accuracy_epochs = np.array(accuracy_epochs)
    epochs = np.array(epochs)
    data = np.vstack((epochs, loss_epochs, accuracy_epochs))

    # Save Data
    name = os.path.join(path_results_network, f"Data_trainning.npy")
    np.save(name, data)

    # Plot Data
    fig11, ax11, ax12 = fancy_plots_2()
    plot_states(
        fig11,
        ax11,
        ax12,
        data[1, :],
        data[2, :],
        data[0, :],
        "tranning",
        path_results_network,
    )
    name_network = os.path.join(path_results_network, f"NN")
    name_onnix = name_network + "." + "onnx"
    torch.onnx.export(
        model,
        images,
        name_onnix,
        input_names=["Image"],
        output_names=["Probabilities"],
        opset_version=11,
    )

    # Show neural network architecture
    netron.start(name_onnix)
    return None


def main():
    """
    Inputs:
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Parser.add_argument(
        "--type",
        default="simple",
        help="Type of neural netwokr",
    )

    Parser.add_argument(
        "--ImagePath",
        default="../CIFAR10",
        help="Base path of images",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    type_NN = Args.type
    Images_path = Args.ImagePath

    # Setup all needed parameters including file reading
    (
        Dirname_train,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainLabels,
        NumClasses,
    ) = SetupAll(Images_path, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        TrainLabels,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        Images_path,
        LogsPath,
        type_NN,
        Dirname_train,
    )


if __name__ == "__main__":
    main()
