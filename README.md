# BFE_Project
## Retina Cataract Detection System

This project provides an offline CustomTkinter-based GUI application for detecting Cataract vs Normal retina images using a trained EfficientNet-based ML model.

The interface allows you to upload any retina fundus image and instantly get:

- Predicted class (Normal / Cataract)
- Confidence percentage (%)
- Preview of the uploaded image
## 🚀 Steps to use the model
### 1️⃣ Download the ZIP File
Click the Code → Download ZIP button on this repository and extract the folder to any location on your PC.

### 2️⃣ Open the Directory in Terminal
Navigate to the extracted folder and:
- Windows: Shift + Right Click → Open PowerShell window here
- Mac/Linux: Right Click → Open in Terminal

### 3️⃣ Install Requirements
Run:
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the GUI Application
```bash
python retina_gui.py
```

### 5️⃣ Upload Test Images
Use images inside test_images/ or any retina image of the same type and resolution.

### 6️⃣ View Results
Left panel: image preview
Right panel: prediction + confidence(%)

### Enjoy using the Retina Cataract Detection System!
## 📊 This is the model's metrics
 Classes: ['cataract', 'normal']
 
Overall metrics on test set:
 Accuracy : 0.8901
 Precision: 0.8913
 Recall   : 0.8913
 F1 score : 0.8913

Classification report:
```
              precision    recall  f1-score   support

    cataract       0.89      0.89      0.89        45
      normal       0.89      0.89      0.89        46

    accuracy                           0.89        91
   macro avg       0.89      0.89      0.89        91
weighted avg       0.89      0.89      0.89        91
```
### Confusion Matrix
<img width="436" height="390" alt="image" src="https://github.com/user-attachments/assets/ac99c241-c3a0-4628-8e78-24f3065d223a" />

### ROC Curve
<img width="536" height="470" alt="image" src="https://github.com/user-attachments/assets/57741d2b-40e3-420a-b8c7-04ce5348e9d3" />

