# 🫁 Ensemble Transfer Learning for Pneumonia Detection

This project implements an **ensemble deep learning model** combining **DenseNet121**, **MobileNet**, and **EfficientNet** for accurate and efficient pneumonia detection using chest X-ray images (CXR). The model is trained using **transfer learning** and achieves a **test accuracy of 99.24%**.

> 📄 [**Research Paper**](./Ensemble_Transfer_Learning_Pneumonia.pdf)  
> 📊 [**Dataset: Chest X-Ray Images (Pneumonia)**](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 🚀 Project Overview

Pneumonia is a serious respiratory condition that demands quick and accurate diagnosis. This work leverages the strengths of three powerful deep learning architectures:

- **DenseNet121** – for efficient gradient flow and feature reuse  
- **MobileNetV3** – for lightweight, mobile-friendly performance  
- **EfficientNetV2S** – for accuracy with fewer parameters

These models are fine-tuned and fused into an **ensemble model** using weighted averaging.

---

## 📁 Dataset

- **Source**: [Kaggle Chest X-ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,856
  - Normal: ~1,583
  - Pneumonia: ~4,273
- **Split**:
  - Training: 4,127
  - Validation: 521
  - Testing: 523
- **Preprocessing**: Resized to 224×224, normalized to [0, 1]

---

## 🧠 Model Architecture

Each model processes the input image independently. Their feature maps are concatenated and passed through a fully connected layer for binary classification (Normal / Pneumonia).

### Training Configuration
- Loss Function: `Binary Crossentropy`
- Optimizer: `Adam`
- Learning Rate: `1e-4` with decay on plateau
- Batch Size: `16`
- Epochs: `Up to 28` (with EarlyStopping)

---

## ✅ Performance

| Metric          | Value     |
|-----------------|-----------|
| **Test Accuracy** | 99.24%    |
| **Test Loss**     | 0.0169    |
| **Precision**      | 98.7%     |
| **Recall**         | 99.3%     |
| **F1 Score**       | 99.0%     |
| **AUC (ROC)**      | 1.00      |

---

## 🎯 Model Evaluation Highlights

- **Confusion Matrix**  
  - TP: 384, TN: 134, FP: 1, FN: 4

- **Explainability with Grad-CAM**  
  - Shows that model focuses correctly on lung regions

- **Misclassification Analysis**  
  - Most false results were due to low-quality X-rays or overlapping symptoms

- **Statistical Significance**  
  - Validated using paired t-tests and confidence intervals

---

## 💡 Key Features

- 🧪 **Transfer Learning** for faster convergence and better generalization  
- 🔗 **Model Ensemble** for robust predictions  
- 🩺 **Medical Imaging Optimized** pipeline  
- 🧠 **Explainable AI (XAI)** via Grad-CAM

---

## 🏥 Real-World Implications

- Assists radiologists in **early diagnosis**
- Supports deployment in **low-resource settings**
- Mobile-friendly models allow **real-time clinical use**

---

## 📌 Future Work

- Integrate other data modalities (e.g. CT scans, clinical records)
- Use attention-based fusion strategies
- Deploy as an edge or web-based diagnostic tool

---

## 👨‍💻 Authors

- **Pragyan Dhungana**
- [**Shatrudhan Chaudhary**](https://github.com/jassatish)
- **Mithu Roy**
- **Rupak Aryal**
- **Mahesh T R** (Advisor)


---
## 👥 Contributors

<table>
  <tr>
    <td align="center"><a href="https://github.com/pragyan127"><img src="https://avatars.githubusercontent.com/pragyan127" width="80px;" alt=""/><br /><sub><b>Pragyan Dhungana</b></sub></a></td>
    <td align="center"><a href="https://github.com/jassatish"><img src="https://avatars.githubusercontent.com/jassatish" width="80px;" alt=""/><br /><sub><b>Shatrudhan Chaudhary</b></sub></a></td>
  </tr>
</table>

## 📄 Citation

If you use this work, please cite it as:

> Dhungana, Pragyan, et al.  
> *Ensemble Deep Learning Approach for Pneumonia Detection Using DenseNet, MobileNet, and EfficientNet with Transfer Learning*.  
> International Conference on Emerging Research in Computational Science, 2025.

### BibTeX:
```bibtex
@inproceedings{dhungana2025ensemble,
  title={Ensemble Deep Learning Approach for Pneumonia Detection Using DenseNet, MobileNet, and EfficientNet with Transfer Learning},
  author={Dhungana, Pragyan and others},
  booktitle={International Conference on Emerging Research in Computational Science},
  year={2025}
}
