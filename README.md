# Explainable Skin Lesion Classification with Attention-Enhanced CNNs

This project presents an **interpretable deep learning framework** for **skin lesion classification** that integrates **trainable, metadata-conditioned spatial attention** directly into the learning process. Unlike post-hoc explainability methods such as Grad-CAM, the proposed model learns **clinically meaningful attention maps during training**, improving transparency and trust while maintaining strong predictive performance.

---

## 🚀 Key Features

- **End-to-End Interpretable CNN**
  - Attention maps are learned *during training*, not generated post-hoc
  - Enables transparent and clinically meaningful decision-making

- **Metadata-Conditioned Spatial Attention**
  - Attention is dynamically guided using patient metadata (age, sex, lesion location)
  - Improves both interpretability and class-balanced performance

- **Efficient Architecture**
  - EfficientNet-B0 backbone pretrained on ImageNet
  - Lightweight attention module with minimal computational overhead

- **Hybrid Training Objective**
  - Cross-entropy loss for classification
  - Attention regularization for stable, non-degenerate focus patterns

- **Comprehensive Evaluation**
  - Comparison with Grad-CAM baseline
  - Ablation studies on attention, metadata conditioning, and regularization

---

## 🧠 Problem Motivation

Deep CNNs have demonstrated strong performance in automated skin lesion classification, but their **black-box nature** limits clinical adoption. Post-hoc interpretability methods (e.g., Grad-CAM) provide explanations only after training and do not influence how models learn.

**Goal of this project:**

> Design a skin lesion classification model that is both **accurate and intrinsically interpretable** by embedding attention mechanisms directly into the training process.

---

## 🛠 Tech Stack

- **Framework**: PyTorch  
- **Backbone Model**: EfficientNet-B0 (ImageNet pretrained)  
- **Attention Mechanism**: Metadata-conditioned spatial attention  
- **Optimization**: AdamW, gradient clipping, learning-rate scheduling  
- **Visualization**: Learned attention heatmaps & Grad-CAM (baseline)

---

## 📊 Dataset

- **Dataset**: HAM10000 (Skin Cancer MNIST)
- **Images**: 10,000 dermoscopic images
- **Classes**: 7 skin lesion categories (benign & malignant)
- **Metadata**:
  - Age (z-score normalized)
  - Sex (one-hot encoded)
  - Lesion localization (one-hot encoded)

### Data Split
- Training: 75%
- Validation: 12.5%
- Test: 12.5% (stratified sampling)

### Preprocessing
- Image resizing to 224×224
- ImageNet normalization
- Data augmentation (rotation, flips, color jitter)

---

## ⚡ Training Setup

- **Baseline Model**
  - EfficientNet-B0 + metadata
  - Cross-entropy loss
  - Grad-CAM for post-hoc explanations

- **Attention-Based Model**
  - Metadata-conditioned spatial attention
  - Hybrid loss:
    ```
    L = L_CE + λ L_attn
    ```
  - Differential learning rates:
    - Backbone: 1e-4
    - Attention + classifier: 1e-3

- **Training Details**
  - AdamW optimizer
  - Gradient clipping
  - Early stopping
  - Learning rate scheduling

---

## 📈 Results & Analysis

### Classification Performance

| Model              | Accuracy | Macro-F1 |
|-------------------|----------|----------|
| Baseline + Grad-CAM | 87.38%   | 0.7765   |
| Full Attention Model | 87.38%   | 0.7889  |

- Attention improves **class-balanced performance**
- Maintains accuracy while improving interpretability

### Interpretability Findings
- Grad-CAM often highlights diffuse or off-center regions
- Learned attention maps are:
  - More localized
  - More stable
  - Clinically meaningful

---

## 🔬 Ablation Studies

| Model Variant       | Accuracy | Macro-F1 |
|--------------------|----------|----------|
| Full Model         | 0.8738   | 0.7889   |
| No Attention       | 0.8802   | 0.7650   |
| No Metadata        | 0.8658   | 0.7627   |
| No Regularization  | 0.8794   | 0.8090   |

### Key Insights
- **Metadata is essential** for meaningful attention maps
- Removing regularization improves F1 but degrades interpretability
- Highlights trade-off between performance and explainability

---

## 🔗 Code & Notebook

- **Kaggle Notebook**:  
  https://www.kaggle.com/code/tejasvenu/final-efficientnet

- **Downloadable Version**:  
  https://drive.google.com/file/d/1ny_iCh5W4L7PjNKKL0VU_pUvNxp_51q1/view

---

## 🏁 Conclusion

This work demonstrates that **interpretability can be embedded directly into CNN training** without sacrificing accuracy. By combining **metadata-conditioned attention** with regularization, the model achieves **transparent, stable, and clinically relevant explanations**, making it more suitable for real-world medical applications.

---



