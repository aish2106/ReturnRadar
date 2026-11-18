# ReturnRadar: E-Commerce Return Risk Prediction and Customer Behavior Analysis

**Predict e-commerce return risk using Machine Learning and understand the factors influencing customer returns.**

## 1. Project Overview

ReturnRadar is a Machine Learning project that predicts whether an e-commerce order item is likely to be returned.

The project focuses on identifying return patterns, analyzing customer and product-related behavior, and building predictive models to support better return-risk assessment.

It combines Exploratory Data Analysis (EDA), data preprocessing, feature engineering, classification algorithms, model evaluation, Explainable AI (SHAP), and an interactive Streamlit application.

## 2. Demo

**Streamlit Application:** http://54.153.165.252:8080

## 3. Problem Statement and Impact

Product returns are a major challenge for e-commerce businesses. Frequent returns can increase operational costs, create inventory management difficulties, and affect business profitability.

ReturnRadar aims to predict potential product returns using historical order and customer-related information.

**Who benefits?**

- **E-commerce businesses:** Identify orders with higher return risk.
- **Business analysts:** Understand patterns in customer return behavior.
- **Inventory teams:** Gain insights that can support inventory planning.
- **Decision-makers:** Use data-driven insights to improve business strategies.

## 4. Dataset

The project uses an e-commerce order-item dataset based on ASOS data.

- **Dataset source:** (https://github.com/davidschulte/bads2021?utm_source=chatgpt.com)
- **Dataset size:** 100,000 rows (project dataset)
- **Target variable:** `return`
- **Problem type:** Binary Classification

**Features used in the project include:**

| Feature | Description |
|---|---|
| item_price | Price of the product |
| item_id | Product identifier |
| age | Customer age |
| order_month | Month of order |
| order_day_of_week | Day of the week when the order was placed |
| days_since_registration | Customer registration duration |
| delivery_days | Delivery duration |

Additional categorical features were transformed during preprocessing.

**Target variable:**

- `0` – Not Returned
- `1` – Returned

## 5. Approach

The project follows a structured Machine Learning workflow.

1. **Data Loading:** Imported the dataset and selected the required records.
2. **Exploratory Data Analysis:** Examined distributions, missing values, target balance, and relationships between variables.
3. **Data Cleaning:** Handled missing values, checked duplicates, and investigated invalid values.
4. **Feature Engineering:** Extracted useful information from date columns and created additional features.
5. **Data Preprocessing:**
   - Numerical features: Imputation and Standard Scaling.
   - Categorical features: Imputation and One-Hot Encoding.
6. **Train-Test Split:** Divided the dataset into training and testing sets.
7. **Model Building:** Implemented multiple classification algorithms.
8. **Model Evaluation:** Compared models using accuracy, precision, recall, F1-score, and ROC-AUC.
9. **Hyperparameter Tuning:** Applied model tuning to improve predictive performance.
10. **Explainable AI:** Used SHAP to understand feature contributions to Random Forest return predictions.
11. **Deployment:** Developed an interactive Streamlit application.

## 6. Model Results

The following table presents the baseline model comparison results.

| Model | Accuracy | Return Recall | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 65.23% | 65% | 0.7153 |
| Decision Tree | 61.73% | 59% | 0.6158 |
| Random Forest | 61.27% | 36% | 0.6983 |
| Gradient Boosting | 62.69% | 71% | 0.6877 |

**Tuned Random Forest – Reported Test Results**

| Metric | Score |
|---|---:|
| Accuracy | 64.04% |
| Precision | 57.95% |
| Recall | 78.46% |
| F1-score | 66.66% |
| ROC-AUC | 0.7099 |

The tuned Random Forest achieved a higher recall for returned orders, helping identify a greater proportion of actual returns.

*Note: Baseline comparison scores and tuned model test scores represent different evaluation stages.*

## 7. Key Insights and Findings

- Product and customer-related features provide useful information for return prediction.
- Classifying returned orders requires attention to recall, not accuracy alone.
- Hyperparameter tuning and class balancing can help improve the detection of returned orders.
- SHAP Feature Importance provides additional insight into which features influence model predictions.
- The project demonstrates how Machine Learning can support e-commerce return-risk analysis.

## 8. Challenges Faced

Some of the challenges encountered during development included:

**1. Missing Values**

Certain dataset columns contained missing values. These were handled using appropriate imputation techniques within the preprocessing pipeline.

**2. Categorical Data Encoding**

Categorical variables required One-Hot Encoding before they could be used by the classification models.

**3. Class Imbalance**

Class balancing was explored using the `class_weight='balanced'` option in Random Forest.

**4. Model Performance**

Initial models did not achieve the expected accuracy. Multiple algorithms and hyperparameter tuning were explored to improve model performance.

**5. SHAP Compatibility**

SHAP initially encountered an issue because the validation data was stored as a SciPy sparse matrix.

The issue was resolved by converting the sparse matrix into a dense NumPy array using `.toarray()`.

**6. SHAP Feature Names**

The initial visualization displayed generic feature numbers. This was resolved by extracting the transformed feature names using `preprocessor.get_feature_names_out()`.

## 9. How to Run Locally

**Prerequisites**

- Python 3.10
- Git
- pip

**Step 1: Clone the repository**

```bash
git clone (https://github.com/aish2106/ReturnRadar.git)
```

**Step 2: Navigate to the project folder**

```bash
cd ReturnRadar
```

**Step 3: Create a virtual environment**

```bash
python -m venv venv
```

**Step 4: Activate the virtual environment (Windows)**

```bash
venv\Scripts\activate
```

**Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 6: Run the Streamlit application**

```bash
streamlit run app.py
```

The application will open in your default browser.

## 10. Limitations and Future Improvements

**Current Limitations**

- Model performance depends on the quality and representativeness of the dataset.
- The dataset may not capture every factor influencing customer return decisions.
- The model's predictions are probabilistic and cannot guarantee whether an order will be returned.
- Model performance may vary when applied to new e-commerce datasets.

**Future Improvements**

- Explore additional customer behavior and product-related features.
- Experiment with advanced classification algorithms.
- Improve model performance through further optimization.
- Expand the dataset with more recent e-commerce records.
- Enhance prediction explainability and model monitoring.

## 11. Tech Stack and Contact

**Technologies Used**

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Application Development | Streamlit |
| Development Environment | Jupyter Notebook |
| Version Control | Git, GitHub |
| Deployment | AWS EC2 |



---

**Thank you for visiting ReturnRadar!**
