# Alohomora

## Phase 1

To execute the algorithms associated with Phase 1 of the homework, follow these steps:

1. **Navigate to the Phase1 Code Directory**

   Open your terminal and change your current directory to `Phase1/Code`:

   ```bash
   cd Phase1/Code
   ```

2. **Run the Wrapper Script**

   Execute the `Wrapper.py` script using Python:

   ```bash
   python3 Wrapper.py
   ```

   This program will run the necessary algorithms and save the resulting images in subdirectories within the `Code` folder. No additional arguments are required to execute this program.

---

## Phase 2

To execute the algorithms associated with Phase 2 of the homework, follow these steps:

1. **Navigate to the Phase2 Code Directory**

   Open your terminal and change your current directory to `Phase2/Code`:

   ```bash
   cd Phase2/Code
   ```

2. **Train a Neural Network Architecture**

   You can select from different neural network architectures to train. For example:

   ### Simple Neural Network Training

   Run the following command:

   ```bash
   python3 Train.py --type simple
   ```

   ### Advanced Neural Network Training

   Run the following command:

   ```bash
   python3 Train.py --type advanced
   ```

   ### ResNet Neural Network Training

   Run the following command:

   ```bash
   python3 Train.py --type resnet
   ```

   ### ResNeXt Neural Network Training

   Run the following command:

   ```bash
   python3 Train.py --type resnext
   ```

   ### DenseNet Neural Network Training

   Run the following command:

   ```bash
   python3 Train.py --type dense
   ```

3. **Test a Neural Network Architecture**

   Before testing a neural network architecture, ensure that it has been trained beforehand. The code relies on the optimized values for each neural network and saves results related to loss, accuracy, and other metrics. The predefined number of epochs is 50; this value should be the same for both training and testing.

   To test the different neural network architectures, run the following commands:

   ```bash
   python3 Test.py --type simple
   python3 Test.py --type advanced
   python3 Test.py --type resnet
   python3 Test.py --type resnext
   python3 Test.py --type dense
   ```

# Computer_vision_HW0
