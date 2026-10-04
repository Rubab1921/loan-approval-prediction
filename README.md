#  Loan Approval Prediction Application

An end-to-end Machine Learning project that predicts whether a loan application will be approved or rejected based on applicant financial and personal details using Logistic Regression and an interactive Gradio web application.

---

##  Project Overview
Loan decisioning processes in financial institutions require quick, objective, and data-driven risk assessments. This application leverages historical loan application data to evaluate applicant risk factors and predict approval probability in real time.

---

##  Dataset Description
The dataset contains applicant demographics, credit history, income details, and loan parameters:
* **Gender, Married, Dependents, Education, Self_Employed:** Applicant background variables.
* **ApplicantIncome & CoapplicantIncome:** Income parameters.
* **LoanAmount & Loan_Amount_Term:** Requested financing structure.
* **Credit_History:** Primary indicator (1.0 = Good, 0.0 = Bad).
* **Property_Area:** Urban, Semiurban, Rural.
* **Loan_Status (Target):** Y (Approved) or N (Rejected).

---

##  Data Preprocessing & Feature Engineering
1. **Missing Value Imputation:**
   * Categorical features imputed using mode.
   * Numerical features (`LoanAmount`, `Loan_Amount_Term`) imputed using median.
2. **Encoding:**
   * Binary mapping for categorical features.
   * One-Hot Encoding (`pd.get_dummies`) applied to `Property_Area`.
3. **Feature Scaling:**
   * `StandardScaler` applied to normalize feature values for convergence stability.

---

##  Model Selection & Training
* **Algorithm:** Logistic Regression (`solver='liblinear'`).
* **Train-Test Split:** 80% Training / 20% Testing (`random_state=42`).
* **Optimization:** Scaled inputs resolved solver convergence issues (`ConvergenceWarning`).

---

##  Model Performance & Evaluation

* **Accuracy:** ~78.86%
* **Precision:** 0.76 (Approved class)
* **Recall:** 0.99 (Approved class)
* **F1-Score:** 0.86

### Confusion Matrix
| | Predicted Rejected (0) | Predicted Approved (1) |
|---|---|---|
| **Actual Rejected (0)** | 18 | 25 |
| **Actual Approved (1)** | 1 | 79 |

---

##  Interactive Web Application (Deployment)
The application is deployed interactively using **Gradio** in Python. Users can select input details and obtain real-time approval predictions with confidence probabilities.

### Test Case Results
1. **Approved Case:** High income, good credit history (1.0) -> **APPROVED ✅** (~92% Probability)
2. **Rejected Case:** Bad credit history (0.0) -> **REJECTED ❌** (~79% Rejection Probability)

---

##  Limitations & Future Improvements
1. **Dataset Size:** Trained on a relatively small dataset (~614 rows); expanding data will improve robustness.
2. **Class Imbalance:** Dataset contains more approved cases than rejected cases, leading to higher recall than precision for approvals.
3. **Future Work:** Try ensemble models like Random Forest or XGBoost and deploy permanently on Hugging Face Spaces or Streamlit Community Cloud.# loan-approval-prediction
