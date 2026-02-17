Task 3 – AI & ML Internship (Elevate Labs)
🎯 Objective

To implement and understand Simple & Multiple Linear Regression using Scikit-learn and evaluate model performance using standard regression metrics.

🛠 Tools & Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

📂 Dataset

The dataset used: housing_data.csv

It contains features such as:

Area

Bedrooms

Bathrooms

Stories

Parking

Furnishing Status

Other categorical features

Target Variable: Price

⚙️ Steps Performed
1️⃣ Data Loading

Loaded dataset using Pandas.

2️⃣ Data Preprocessing

Handled missing values

Converted categorical variables using One-Hot Encoding

Selected numerical features for modeling

3️⃣ Train-Test Split

Split dataset into:

80% Training Data

20% Testing Data

4️⃣ Model Building

Implemented Multiple Linear Regression using:

from sklearn.linear_model import LinearRegression

5️⃣ Model Evaluation

Evaluated model performance using:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

R² Score

6️⃣ Visualization

Plotted Actual vs Predicted Prices to analyze model performance.

📊 Model Evaluation Metrics

MAE → Measures average absolute prediction error

MSE → Penalizes larger errors more heavily

R² Score → Indicates how well the model explains variance

📈 Output

Generated:

actual_vs_predicted.png
→ Visual comparison of predicted and actual house prices.

The graph shows a strong positive correlation, indicating that the model captures pricing trends effectively.

🧠 Key Learnings

Understanding regression assumptions

Interpreting coefficients

Handling categorical data in regression

Evaluating model performance

Importance of preprocessing before modeling

🚀 Future Improvements

Apply feature scaling

Try Ridge & Lasso Regression

Perform cross-validation

Compare with other regression algorithms

📌 Conclusion

Linear Regression provides a strong baseline model for price prediction problems.
Proper preprocessing and feature engineering significantly improve model performance.
