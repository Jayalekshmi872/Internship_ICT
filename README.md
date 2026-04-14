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

🔹 .Feature engineering 

created 3 tires from location like tiers1,tiers2,tiers3

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

applied logtransform on avg_cost_for_two

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

🚀 Week 3 – Model Training & Evaluation

🔹 1. Models Implemented

The following machine learning models were trained and evaluated:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Extra Trees  

---

🔹 2. Model Evaluation Metrics

Models were evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  

A comparison table was created to analyze model performance.

---

🔹 3. Scaling Comparison

Scaling was tested to understand its impact on model performance.

- Scaling improved performance for:
  - Logistic Regression  
  - SVM  
  - KNN  

- Scaling did NOT improve performance for:
  - Random Forest  
  - Decision Tree  
  - Gradient Boosting  

📌 Observation:
Random Forest performed better without scaling, as tree-based models are not sensitive to feature scaling.

---

🔹 4. Hyperparameter Tuning

Hyperparameter tuning was applied to Random Forest using GridSearchCV.

- Slight improvement in Accuracy  
- Slight decrease in ROC-AUC  

📌 Final Decision:
Untuned Random Forest was selected as it achieved better ROC-AUC score.

---

🔹 5. Final Model Selection

🏆 **Random Forest (without scaling, without tuning)** was selected as the final model.

📊 Performance:
- Accuracy: ~0.72  
- ROC-AUC: ~0.80  

📌 Reason:
- Highest ROC-AUC score  
- Stable performance  
- Suitable for dataset  

---

🔹 6. Feature Importance Analysis

Feature importance was extracted using Random Forest.

- Identified top contributing features  
- Visualized top features using bar plot  

📌 Insight:
Certain features like  avg_cost_for_two,north ,location_tier options significantly influence restaurant success.

---

🔹 7. Model Saving

Final model was saved using pickle:

import pickle   # for saving it

with open('tuned_random_forest_zomato.pkl','wb') as f:
  pickle.dump(rf1,f)

  
📌 Author

- Jayalekshmi
