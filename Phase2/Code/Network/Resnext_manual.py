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

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    loss = F.cross_entropy(out, labels)
    return loss


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = loss_fn(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = loss_fn(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"loss": loss.detach(), "acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"loss": epoch_loss.item(), "acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(
                epoch, result["loss"], result["acc"]
            )
        )


class ResnextBasic(ImageClassificationBase):
    def __init__(self, InputSize, OutputSize):
        super(ResnextBasic, self).__init__()

        # Input
        self.conv1 = nn.Conv2d(InputSize[0], 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Firs block

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # First Block pararell
        self.conv3_p = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3_p = nn.BatchNorm2d(64)

        self.conv4_p = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4_p = nn.BatchNorm2d(64)

        ## second block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        ## second block pararell
        self.conv5_p = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5_p = nn.BatchNorm2d(128)

        self.conv7_p = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7_p = nn.BatchNorm2d(128)

        # aux convolution to normalize the interconexion
        self.conv_aux = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_aux = nn.BatchNorm2d(128)

        ## third block
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(256)

        # aux convolution to normalize the interconexion
        self.conv_aux_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn_aux_2 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Added adaptive pooling

        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, OutputSize)

        self.dropout = nn.Dropout(0.5)

        self.name = "Resnext"

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Save variable for later
        x_input_1 = x
        x_input_1_p = x

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x_input_1_p = self.conv3_p(x_input_1_p)
        x_input_1_p = self.bn3_p(x_input_1_p)
        x_input_1_p = F.relu(x_input_1_p)
        x_input_1_p = self.conv4_p(x_input_1_p)
        x_input_1_p = self.bn4_p(x_input_1_p)
        x_input_1_p = F.relu(x_input_1_p)

        x = x + x_input_1 + x_input_1_p

        x_input_2 = x
        x_input_2_p = x

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)

        x_input_2_p = self.conv5_p(x_input_2_p)
        x_input_2_p = self.bn5_p(x_input_2_p)
        x_input_2_p = F.relu(x_input_2_p)
        x_input_2_p = self.conv7_p(x_input_2_p)
        x_input_2_p = self.bn7_p(x_input_2_p)
        x_input_2_p = F.relu(x_input_2_p)

        x_normalize = self.conv_aux(x_input_2)
        x_normalize = self.bn_aux(x_normalize)

        x = x + x_normalize + x_input_2_p

        x_input_3 = x

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)

        x_normalize_2 = self.conv_aux_2(x_input_3)
        x_normalize_2 = self.bn_aux_2(x_normalize_2)

        x = x + x_normalize_2
        x = self.adaptive_pool(x)  # Output shape: [batch_size, 256, 1, 1]

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x
