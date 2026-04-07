🍽️ Restaurant Success Prediction (FoodTech Analytics)

📌 Project Overview

This project aims to analyze restaurant data and predict restaurant success using machine learning techniques. The workflow includes data scraping, exploratory data analysis (EDA), preprocessing, and feature engineering.

---

📊 Dataset Columns

Initial dataset contains:

- "name" – Restaurant name
- "url" – Source link
- "rating" – Customer rating
- "location" – Combined city + area
- "city" – City name
- "area" – Area name
- "cuisine" – Type of food
- "price" – Price category
- "reviews" – Number of reviews
- "restaurant_type" – Type (Cafe, Fast Food, etc.)
- "num_ratings" – Total ratings count
- "online_delivery" – Delivery availability
- "table_booking" – Booking availability
- "avg_cost_for_two" – Average cost for two people
- "is_delivering_now" – Current delivery status
- "switch_to_order_menu" – Order menu option

---

📊 Week 1 – Exploratory Data Analysis (EDA)

🔹 Univariate Analysis

- Distribution of "rating"
- Distribution of "avg_cost_for_two"
- Count of "restaurant_type"

---

🔹 Bivariate Analysis

- "rating" vs "restaurant_type"
- "rating" vs "price"
- "avg_cost_for_two" vs "rating"

---

🔹 Data Cleaning

- Removed duplicate rows using:

df.drop_duplicates(inplace=True)

---

🔹 Missing Value Analysis

- Identified missing values in:
  - "rating"
  - "cuisine"
  - "price"
  - "avg_cost_for_two"

---

⚙️ Week 2 – Data Preprocessing

🔹 1. Handling Missing Values

Used groupby-based filling strategy:

- "rating" → filled using mean/median based on groups
- "cuisine" → filled using mode
- "price" → filled using related features
- "avg_cost_for_two" → filled using grouped mean

---

🔹 2. Encoding

✅ Binary Encoding (0 / 1)

Applied mapping:

df['online_delivery'] = df['online_delivery'].map({'Yes': 1, 'No': 0})
df['table_booking'] = df['table_booking'].map({'Yes': 1, 'No': 0})

---

🔹 3. Feature Scaling

Applied StandardScaler on numerical feature:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['num_ratings']] = scaler.fit_transform(df[['num_ratings']])

---

🔹 4. Target Variable Creation

Converted rating into classification target:

df['success'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

- 1 → Successful restaurant
- 0 → Not successful

---

🔹 5. Avoiding Data Leakage

After creating target:

df.drop('rating', axis=1, inplace=True)

✔ Prevents model from cheating
✔ Ensures proper learning

---

🎯 Problem Type

- Classification Problem
- Goal: Predict whether a restaurant is successful or not

---

📦 Final Dataset

- Cleaned dataset
- Missing values handled
- Encoded features
- Scaled numerical data
- Target variable ("success") created

---

🚀 Next Steps

- Model training (Logistic Regression, Random Forest, etc.)
- Hyperparameter tuning
- Model evaluation (Accuracy, Precision, Recall, F1-score)

---

🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn

---

📌 Author

- Jayalekshmi
