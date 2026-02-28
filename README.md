# Linear Regression from Scratch (NumPy Implementation)

##  Overview

This project implements **Linear Regression using Gradient Descent from scratch** without relying on scikit-learn or any built-in machine learning models.

The goal of this project is to deeply understand:

- How Linear Regression works mathematically
- How Gradient Descent updates parameters
- How feature normalization affects convergence
- How evaluation metrics like R² are computed
- How to structure ML code in a clean and professional way

This repository focuses on building strong ML fundamentals rather than using high-level abstractions.

---

##  What Is Implemented

- Feature standardization (mean normalization)
- Bias term handling
- Fully vectorized Gradient Descent
- Custom train-test split function
- Custom R² score implementation
- Clean separation of model, utilities, and training pipeline

No machine learning libraries were used for model training.

---

##  Mathematical Formulation

The model minimizes the Mean Squared Error:

J(w) = (1 / 2m) * Σ (Xw − y)²

Gradient Descent update rule:

w := w − α * (1/m) * Xᵀ (Xw − y)

Where:
- α = learning rate  
- m = number of training samples  
- X = feature matrix  
- w = weight vector  

---

##  Dataset

The model is evaluated on a housing dataset with features such as:

- RM (Average number of rooms)
- LSTAT (% lower status population)
- PTRATIO (Pupil-teacher ratio)
- MEDV (Median home value)

---

##  Results

Test Set R² Score: ~0.76

This means the model explains approximately 76% of the variance in housing prices using a linear model.

---

##  Project Structure
linear-regression-from-scratch/
│
├── data/
│ └── housing.csv
├── model.py # Linear Regression implementation
├── utils.py # Train-test split and R² score
├── train.py # Training pipeline
├── README.md
└── requirements.txt

## 🔮 Possible Extensions

While this project focuses on core linear regression fundamentals, the framework can be extended to explore:

- Polynomial feature expansion
- L2 Regularization (Ridge Regression)
- Implementation using the Normal Equation
- Logistic Regression from scratch
- Support Vector Machines (SVM)
- Modular optimizer implementations (Momentum, Adam)

These extensions would further strengthen understanding of optimization and model generalization.

---

##  How to Run

1️⃣ Install dependencies:

pip install -r requirements.txt

2️⃣ Run training:

python train.py

---

##  Why This Project Matters

Understanding machine learning at the implementation level builds clarity that abstraction often hides.

This project demonstrates foundational knowledge in:

- Linear algebra
- Optimization
- Model evaluation
- Clean ML project structuring

Built as part of a structured journey toward Machine Learning engineering.
