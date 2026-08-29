# AIShield – Image Tampering Detection using Deep Learning

AIShield is a deep learning project developed to detect whether an image is authentic or tampered.

The project uses an EfficientNet-based model for binary image classification and Grad-CAM for visual explainability, so the system not only gives a prediction but also highlights the image regions that influenced the model.

## Features

- Classifies images as **Clean** or **Tampered**
- Uses **EfficientNet-Lite0**
- Handles class imbalance using weighted loss
- Generates **Grad-CAM heatmaps**
- Includes model evaluation and visualization
- Provides an interactive **Streamlit application**

## Dataset

The project was developed using the **CASIA Image Tampering Dataset**.

The dataset itself is not included in this repository because of its size.

The dataset splitting script is available at:

`scripts/split_casia.py`

## Model

The current model used for training and inference is:

`EfficientNetLite0Tamper`

located in:

`models/efficientnet_lite0.py`

The trained model is stored in:

`saved_models/efficient_lite0.pth`

Training uses:

- Adam optimizer
- Learning rate: `1e-4`
- Batch size: `16`
- Weighted CrossEntropyLoss
- ReduceLROnPlateau scheduler

## Explainability

Grad-CAM is used to visualize which parts of an image contributed most to the model's prediction.

The project also contains utilities for:

- Error Level Analysis
- Noise residual analysis
- Prediction visualization

## Results

The saved evaluation report contains results on **1,912 test images**.

| Metric | Score |
|---|---:|
| Accuracy | 82% |
| Macro F1-score | 0.77 |
| Weighted F1-score | 0.83 |

Additional results such as the confusion matrix, ROC curve, loss curves, and sample predictions are available in the `results/` folder.

## Streamlit Application

Run the application using:

```bash
streamlit run streamlit_app.py
```

The application allows a user to upload an image and displays:

- Predicted class
- Confidence score
- Clean/tampered probabilities
- Grad-CAM visualization

## Project Structure
```text
AAIAIShieldProject/
├── aishield_main.py
├── train_efficientnet.py
├── streamlit_app.py
├── models/
├── utils/
├── scripts/
├── saved_models/
└── results/
```

## Installation

```bash
git clone https://github.com/mahera2222/AAIAIShieldProject.git
cd AAIAIShieldProject
pip install -r requirements.txt
```

## Technologies Used

Python, PyTorch, Torchvision, EfficientNet, NumPy, Matplotlib, Scikit-learn, Streamlit and Grad-CAM.

## Author

Mahera Sultana Shaik  
M.S. Applied Artificial Intelligence  
Stevens Institute of Technology
