#**DeepFake Detection using CNN**

📌 **Project Overview:**
This project focuses on detecting deepfake images using a high-capacity Convolutional Neural Network (CNN). The model is designed to distinguish between real and fake faces using advanced deep learning techniques.

📂 **Dataset:**
The dataset consists of real and fake face images categorized into training, validation, and testing sets. Data augmentation techniques are applied to improve the model's generalization.

🏗️ **Model Architecture:**
The CNN model follows a structured architecture with multiple convolutional layers, batch normalization, and fully connected layers. Key features include:

Input Shape: (224, 224, 3)
Conv Layers:
      Conv2D (32 filters, kernel size 3x3) → BatchNorm → MaxPooling
      Conv2D (64 filters, kernel size 3x3) → BatchNorm → MaxPooling
      Conv2D (128 filters, kernel size 3x3) → BatchNorm → MaxPooling
Flatten Layer: Converts feature maps to a single vector
Dense Layers:
      Fully Connected Layer (512 neurons) → BatchNorm → Dropout
      Output Layer (1 neuron for binary classification)
Activation Function: Leaky ReLU for convolutional layers
Optimizer: SGD with Cyclical Learning Rate Decay
Learning Rate: Ranges from 1e-5 to 1e-3 with a step size of 1000
Momentum: 0.9 with Nesterov acceleration
Loss Function: Binary Cross-Entropy
Total Parameters: 44,399,553 (169.37 MB)
Trainable Parameters: 44,398,081
Non-Trainable Parameters: 1,472

🏋️‍♂️ **Training:**
The model is trained using:
Epochs: 10 (Early stopping if validation loss does not improve for 30 consecutive epochs) For higher accuracy train the model at least for 1000 epochs.
Callbacks: Checkpointing to save the best model, early stopping to prevent overfitting.

🚀 **Results:**
Evaluation: Accuracy and loss metrics are used to assess model performance.
Testing: The best model is evaluated on a separate test dataset.

🔧 **Usage:**
Clone the repository
Install required dependencies using **pip install -r requirements.txt**
Use **predict.py** to test on new images

📜 **Acknowledgments:**
This project utilizes TensorFlow and Keras for deep learning. The dataset is sourced from publicly available deepfake datasets.
Datasets used:
  1.https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/data

My trained model:https://drive.google.com/file/d/1-1pIXHk7pISX4LjYuVdKRpxXI2UDmFfu/view?usp=drive_link
