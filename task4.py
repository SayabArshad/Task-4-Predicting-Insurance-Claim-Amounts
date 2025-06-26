import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load the dataset
df = pd.read_csv("d:/python_ka_chilla/Internship/task 4/insurance.csv")
print(df.head())
print(df.info())

# Step 2: Encode categorical features using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 3: Define features (X) and target (y)
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test)

# Step 7: Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Step 8: Visualizations

# Age vs Charges
sns.scatterplot(data=df, x='age', y='charges', hue='smoker')
plt.title("Charges vs Age (Colored by Smoking Status)")
plt.show()

# BMI vs Charges
sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker')
plt.title("Charges vs BMI (Colored by Smoking Status)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
