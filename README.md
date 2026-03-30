# 🎓 Student Performance Prediction
### Logistic Regression from Scratch · IOE Pulchowk Campus

A machine learning project that predicts whether a student will **Pass or Fail** based on 21 academic and behavioral features. The core algorithm — Logistic Regression — is implemented entirely from scratch using only NumPy, without any sklearn training functions.

---

## 📁 Project Structure

```
project/
├── app.py                                    # Streamlit web app
├── data_processing.ipynb                     # EDA and data cleaning
├── logistics_regression.ipynb                # Model training from scratch
├── cleaned_student_performance_dataset.csv   # Cleaned dataset
├── X_features.csv                            # Feature matrix
├── y_target.csv                              # Target labels
├── logistic_weights.npy                      # Trained theta weights
├── scaler_mean.npy                           # Feature means for scaling
└── scaler_std.npy                            # Feature stds for scaling
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Total records | 300,000 students |
| Features | 21 (numerical + categorical) |
| Target | Pass / Fail |
| Class distribution | 60.24% Pass, 39.76% Fail |

**Key features:** study hours, attendance, previous grade, assignments completed, practice tests, motivation level, sleep hours, screen time, social media hours, family income, parent education, and more.

---

## 🧮 The Math — Logistic Regression from Scratch

### 1. Why Logistic Regression?

Linear Regression predicts continuous values like 3.5 or -1.2, which makes no sense for a Pass/Fail prediction. We need a **probability between 0 and 1**.

Logistic Regression solves this by applying the **sigmoid function** to the linear combination of features.

---

### 2. Sigmoid Function

The sigmoid function squashes any real number into the range (0, 1):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:
- z < 0 → output closer to 0 (likely Fail)
- z = 0 → output = 0.5 (uncertain)
- z > 0 → output closer to 1 (likely Pass)

In code:
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

---

### 3. Hypothesis Function

For a student with feature vector **x** and weight vector **θ**:

$$z = \theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

$$h_\theta(x) = \sigma(z) = \frac{1}{1 + e^{-\theta^T x}}$$

This gives the **probability that the student will pass**.

- If h(x) ≥ 0.5 → Predict **PASS**
- If h(x) < 0.5 → Predict **FAIL**

The bias term θ₀ is handled by prepending a column of 1s to X:
```python
X = np.c_[np.ones(m), X]   # adds bias column
```

---

### 4. Cost Function — Binary Cross Entropy

We cannot use Mean Squared Error (MSE) for classification because combining MSE with the sigmoid gives a **non-convex** function — gradient descent would get stuck in local minima.

Instead we use **Binary Cross Entropy (Log Loss)**:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

Where:
- m = number of training examples
- y = actual label (1 = Pass, 0 = Fail)
- h = predicted probability

**Intuition:**
- If y=1 and h→1: loss → 0 (correct, no penalty)
- If y=1 and h→0: loss → ∞ (wrong, large penalty)
- If y=0 and h→0: loss → 0 (correct, no penalty)
- If y=0 and h→1: loss → ∞ (wrong, large penalty)

With **L2 Regularization** to prevent overfitting:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h) + (1-y^{(i)}) \log(1-h) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

Note: we do **not** regularize θ₀ (the bias term).

In code:
```python
def compute_cost(X, y, theta, lambda_reg=0.01):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1/m) * np.sum(y * np.log(h + 1e-10) + (1-y) * np.log(1-h + 1e-10))
    cost += (lambda_reg / (2*m)) * np.sum(theta[1:] ** 2)
    return cost
```

---

### 5. Gradient Descent

To minimize J(θ), we compute the gradient and update weights iteratively:

$$\frac{\partial J}{\partial \theta} = \frac{1}{m} X^T (h - y)$$

$$\theta := \theta - \alpha \cdot \frac{\partial J}{\partial \theta}$$

Where α is the **learning rate** (we used α = 0.1).

With regularization (not applied to θ₀):

$$\theta_j := \theta_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m}(h^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j \right] \quad \text{for } j \geq 1$$

In code:
```python
def gradient_descent(X, y, theta, learning_rate=0.1, iterations=2000, lambda_reg=0.01):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        gradient[1:] += (lambda_reg / m) * theta[1:]   # regularization
        theta = theta - learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta, lambda_reg))
    return theta, cost_history
```

**Cost over iterations:**
- Iteration 0 → Cost: 0.6797
- Iteration 200 → Cost: 0.3158
- Iteration 1000 → Cost: 0.2841
- Iteration 2000 → Cost: 0.2829 ✅ Converged

---

### 6. Feature Scaling

Before training, features are standardized so gradient descent converges faster:

$$x' = \frac{x - \mu}{\sigma}$$

Where μ is the mean and σ is the standard deviation of each feature in the training set. The same μ and σ are saved and applied to new inputs during prediction.

---

### 7. One-Hot Encoding

Categorical features (gender, school type, etc.) are converted to binary columns using one-hot encoding with `drop_first=True` to avoid multicollinearity (the dummy variable trap).

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.8694 |
| Precision | 0.8831 |
| Recall | 0.9027 |
| F1 Score | 0.8928 |
| ROC-AUC | 0.9471 |

✅ Our from-scratch implementation matches `sklearn.linear_model.LogisticRegression` exactly at **0.8694 accuracy**, confirming the correctness of our gradient descent and sigmoid implementation.

---

## 🖥️ Streamlit App

The web app allows interactive prediction with a report card style output.

```bash
streamlit run app.py
```

**Features:**
- 21 input sliders and dropdowns
- Pass/Fail report card with probability
- Personalized tips based on weak inputs

---

## 🛠️ Requirements

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

---

## 👥 Team

IOE Pulchowk Campus — AI Project · 6th Semester