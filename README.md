# Customer Credit Score Classification

This project compares various machine learning classification algorithms to predict a customer’s credit score category—Good, Standard, or Poor—based on demographic and financial attributes.

## 🧠 Objective

To evaluate and compare the performance of different classification algorithms for credit score prediction using a cleaned and preprocessed dataset of over 100,000 customers.

## 🚀 Tools & Technologies

- **Python**
- **Pandas**, **NumPy** – Data manipulation and preprocessing
- **Scikit-learn** – ML models (Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes)
- **XGBoost** – High-performance gradient boosting classifier
- **Matplotlib**, **Seaborn** – Data visualization
- **GridSearchCV** – Hyperparameter tuning
- **PCA** – Dimensionality reduction
- **Joblib** – Model saving
- **Google Colab** – Development environment

## 📊 Data Preprocessing

- Imputation of missing values (mean/median)
- One-hot encoding of categorical variables
- Feature scaling using StandardScaler
- Feature selection using PCA and correlation matrix

## 📈 Model Comparison

| Algorithm           | Accuracy (%) |
|---------------------|--------------|
| Logistic Regression | ~62.8        |
| Decision Tree       | ~66.0        |
| Random Forest       | ~68.5        |
| K-Nearest Neighbors | ~65.2        |
| Naive Bayes         | ~60.0        |
| **XGBoost**         | **71.5**     |

> 📌 **XGBoost** achieved the best performance, with an accuracy of **71.5%** and strong generalization across classes.

## ✅ Key Highlights

- Handled a real-world credit score dataset with 100K+ records.
- Built and evaluated 6 classification models using accuracy, confusion matrix, and visualization.
- Deployed best model using `joblib` for reuse and integration.
- Applied PCA for dimensionality reduction and variance preservation.

## 📁 Project Structure
├── Credit_score_classification.ipynb
├── credit_train.csv
├── credit_test.csv
├── model.pkl
└── README.md

## 📌 Future Improvements

- Use of SMOTE to balance class distribution
- Deployment using Streamlit or Flask
- Further hyperparameter optimization for top models

---

## 📬 Contact

For questions or collaboration:  
📧 rithwik.g.misc@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/rithwik-gilla)  
