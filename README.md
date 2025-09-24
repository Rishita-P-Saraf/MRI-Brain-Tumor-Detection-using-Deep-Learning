# 🧠 MRI Brain Tumor Detection using Deep Learning

This project implements a Convolutional Neural Network (CNN) using PyTorch to detect brain tumors from MRI images. The model classifies MRI scans as either `Tumor` or `Healthy`.

---

## 📂 Dataset

- **Source**: [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Classes**:  
  - `yes/`: MRI images with tumors  
  - `no/`: MRI images without tumors  

Images are resized to **128x128 pixels** for consistency.

---

## 📦 Dependencies

- Python 3.x  
- PyTorch  
- NumPy  
- OpenCV  
- Matplotlib  
- Seaborn  
- scikit-learn  

Install dependencies using:

```bash
pip install torch torchvision opencv-python numpy matplotlib seaborn scikit-learn
```
---

## 🛠️ Project Structure
```yaml
brain_tumor_detection/
│
├── data/
│   └── brain_tumor_dataset/
│       ├── yes/
│       └── no/
│
├── main.py
├── utils.py
├── model.py
├── README.md

```
---

## 🧠 Model Architecture
A custom CNN model with the following structure:
```python
Conv2d(3 → 6, kernel_size=5) → Tanh → AvgPool2d
Conv2d(6 → 16, kernel_size=5) → Tanh → AvgPool2d
Flatten → Linear(256 → 120) → Tanh → Linear(120 → 84) → Tanh → Linear(84 → 1) → Sigmoid
```
---

## 🔍 Key Features
- Custom PyTorch Dataset class for preprocessing and loading MRI images
- CNN-based binary classification model
- Training with Adam Optimizer and Binary Cross Entropy Loss
- Model evaluation using confusion matrix, accuracy score, and feature maps
- Train-validation split to monitor overfitting

---

## 📊 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Loss Plot over Epochs
- Model Performance on GPU (if available)

Example evaluation output:
```text
Train Epoch: 400    Loss: 0.007721
Final Accuracy: 100.00%
```
---

## 📈 Visualizations
- Sample MRI images (Tumor & Healthy)
- Confusion matrix
- Output probabilities
- Loss curve (Training vs Validation)
- Feature maps from convolutional layers
  
---

## 🧪 How to Run
1. Clone the repository and place the dataset in ./data/brain_tumor_dataset/
2. Run the main script:
```bash
python main.py
```
3. To visualize feature maps, ensure GPU support is enabled.

---

## 💡 Future Improvements
- Use data augmentation for better generalization
- Try deeper or pre-trained CNN architectures (e.g., ResNet, VGG)
- Integrate with a web interface (e.g., Flask, Streamlit)

---

## 🤝 Acknowledgments
- MLDawn YouTube Channel – for the excellent step-by-step tutorial series on building a Brain Tumor Detector using PyTorch and MRI images.
- Navoneel Chakrabarty on Kaggle – for the MRI Brain Tumor Detection dataset. 

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
