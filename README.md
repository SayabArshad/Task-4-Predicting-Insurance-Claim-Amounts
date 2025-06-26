# Task 4: Insurance Claim Amount Prediction

## 🎯 Objective

The goal of this task is to predict the **insurance charges** based on individual attributes such as age, BMI, smoking status, and more. This is a regression problem where we aim to estimate the medical insurance cost using linear regression.

---

## 📁 Dataset Description

- **Dataset Name:** `insurance.csv`
- **Source:** Local file (path: `task 4/insurance.csv`)
- **Target Variable:** `charges` (insurance cost in USD)
- **Features:**
  - `age`, `sex`, `bmi`, `children`, `smoker`, `region`

---

## ⚙️ Tools & Libraries Used

- Python
- pandas
- seaborn
- matplotlib
- scikit-learn

---

## 🧪 Approach

### 1. Load Dataset
- Used `pandas` to load the CSV file
- Performed initial inspection using `.head()` and `.info()`

### 2. Data Preprocessing
- Applied **one-hot encoding** to categorical variables (`sex`, `smoker`, `region`)
- Removed one category using `drop_first=True` to avoid multicollinearity

### 3. Feature Selection
- **Features (`X`):** all columns except `charges`
- **Target (`y`):** `charges`

### 4. Train-Test Split
- Split data using an 80-20 ratio with `train_test_split`

### 5. Model Training
- Trained a **Linear Regression** model using `scikit-learn`

### 6. Model Evaluation
- Metrics used:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**

### 7. Data Visualization
- **Scatter plots:**
  - `age` vs `charges` (colored by `smoker`)
  - `bmi` vs `charges` (colored by `smoker`)
- **Correlation Heatmap** for all numerical and encoded features

---

## 📊 Results & Insights

- **MAE (Mean Absolute Error):** ~2711.10 USD
- **RMSE (Root Mean Squared Error):** ~5903.15 USD  
  *(Note: Actual values may vary depending on train/test split)*

**Key Insights:**
- Smokers are charged significantly higher than non-smokers.
- Insurance charges increase with age and BMI, especially for smokers.
- Linear regression provides a baseline model, though non-linear models could offer improvements.

---

## 📂 Project Files

- `Task-04.ipynb` – Jupyter notebook with all steps, code, and plots.
- `README.md` – Documentation and explanation of the task.

---

## ✅ Submission Checklist

- [x] Dataset loaded and explored
- [x] Categorical data encoded properly
- [x] Model trained using linear regression
- [x] Model evaluated with MAE and RMSE
- [x] Visualizations created
- [x] Clean, well-commented code
- [x] README.md added
- [x] Uploaded to GitHub
- [x] Submitted on Google Classroom

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SayabArshad/Task-4-Predicting-Insurance-Claim-Amounts.git
