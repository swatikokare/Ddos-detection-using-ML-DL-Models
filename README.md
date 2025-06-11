# Ddos-detection-using-ML-DL-Models
# DDoS Detection using Machine Learning and Deep Learning

This project presents a comprehensive system for detecting Distributed Denial-of-Service (DDoS) attacks using a variety of machine learning and deep learning models. The approach includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

---

## 🚀 Features

- Data preprocessing on a dataset with 100,000+ rows and 20+ features
- Heatmap for missing values
- Feature distribution analysis: numerical, categorical, and continuous
- Visualizations for attack types and protocols
- Baseline ML classifiers:  
  - DNN  
  - KNN  
  - SVM  
  - Decision Tree  
  - Naive Bayes  
  - Quadratic Discriminant Analysis (QDA)  
  - SGD  
  - Logistic Regression  
  - XGBoost  
- Model training and performance evaluation
- Visualizations: Loss vs Epochs, Accuracy vs Epochs (for DNN)
- Accuracy comparison of all classifiers

---

## 📁 Project Structure
├── data/ # Dataset folder
├── notebooks/ # Jupyter notebooks for each step
├── models/ # Saved model files (optional)
├── plots/ # Output figures
├── README.md # Project documentation
└── requirements.txt # Python dependencies


---

## 🛠️ Functional Requirements

- Load and preprocess large-scale datasets (≥100,000 records)
- Analyze and visualize data distributions and outliers
- Normalize and split features
- Train multiple classifiers and evaluate their performance
- Visualize training progress and classification accuracy

---

## 💻 Hardware Requirements

- **CPU**: Intel Core i5/i7/i9 or AMD Ryzen 5/7 (≥ 4 cores)  
- **RAM**: Minimum 8 GB (16 GB recommended)  
- **Storage**: SSD with at least 10 GB of free space  
- **GPU (Optional)**: NVIDIA CUDA-enabled GPU (e.g., GTX 1650+)  
- **Display**: 1920×1080 resolution or higher

---

## 🧰 Software Requirements

- **OS**: Windows, Ubuntu (recommended), or macOS  
- **Python**: Version 3.8 or higher  
- **IDE**: Jupyter Notebook / VS Code / PyCharm / Google Colab  
- **Package Manager**: `pip` or `conda`

## 📊 Output and Evaluation
All models are trained on normalized features.

DNN model includes tracking of training loss and accuracy per epoch.

Model comparison chart shows performance of all classifiers.

Evaluation metrics: Accuracy, Precision, Recall, F1-Score.

---
### 🔌Optional Tools
• Google Colab – For free GPU access in the cloud
• TensorBoard – For DNN training visualization
• Anaconda – For managing Python environments

### Required Python Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
```

### 🧾 Conclusion
This project successfully demonstrates the application of traditional machine learning and deep learning techniques for DDoS attack detection using a structured dataset. Through comprehensive data preprocessing, exploratory analysis, and evaluation of multiple classifiers, we established a reliable detection pipeline. The comparison of various models—ranging from Logistic Regression and SVM to advanced classifiers like XGBoost and DNN—highlights the strengths and trade-offs in terms of accuracy, computational complexity, and interpretability.

The results show that integrating both statistical and deep learning methods can improve detection performance and robustness. This solution provides a foundational framework for further research and development in network intrusion detection, especially in scalable and real-time systems such as those used in Software Defined Networks (SDNs) or cloud infrastructures.

### Future work may include:

Deployment in real-time SDN environments

Use of Transformer-based architectures for temporal flow analysis

Implementation of unsupervised anomaly detection methods using Autoencoders or GANs



