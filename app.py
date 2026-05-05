from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

#if __name__ == '__main__':
 #   app.run(debug=True)

# ✅ Load everything separately
model = pickle.load(open("models/random_forest_model_zomato.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
ohe = pickle.load(open("models/ohe.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))


# 🔹 City → Tier function (same as training)
def get_tier(city):
    city = city.lower()

    tier_1 = ['delhi', 'bangalore', 'mumbai', 'chennai', 'kolkata', 'hyderabad', 'pune']
    tier_2 = ['kochi', 'chandigarh', 'lucknow', 'coimbatore', 'bhopal', 'nagpur']
    
    if city in tier_1:
         return 3
    elif city in tier_2:
         return 2
    else:
         return 1


# 🔹 Home page
@app.route('/')
def home():
    return render_template('index.html')


# 🔹 Prediction
@app.route('/predict', methods=['POST'])
def predict():

    # 🔹 Get inputs
    num_ratings = float(request.form['num_ratings'])
    online_delivery = int(request.form['online_delivery'])
    table_booking = int(request.form['table_booking'])
    avg_cost = float(request.form['avg_cost_for_two'])
    price = int(request.form['price'])

    cuisine_list = request.form.getlist('cuisine')
    cuisine = " ".join(cuisine_list)
    restaurant_type = request.form['restaurant_type']
    city = request.form['city']
    if len(cuisine_list) == 0 :
        cuisine = "unknown"
    # 🔹 Convert city → tier
    location_tier = get_tier(city)

    # 🔹 Transform cuisine (TF-IDF)
    cuisine_vec = tfidf.transform([cuisine]).toarray()

    # 🔹 Encode restaurant type (OHE)
    rest_vec = ohe.transform([[restaurant_type]])
    # 🔹 Scale numeric features
    num_scaled = scaler.transform([[num_ratings]])

    # 🔹 Combine ALL features
    features = np.concatenate([
        num_scaled,
        [[online_delivery]],
         [[table_booking]] ,
           [[avg_cost]],
           cuisine_vec,
             [[price]],
           [[location_tier ]],
        rest_vec,
               
    ], axis=1)

    # 🔹 Predict using MODEL (correct)
    proba = model.predict_proba(features)[0][1]

    prediction = 1 if proba > 0.5 else 0

    return redirect(url_for('result', pred=int(prediction),prob = float(proba) ,online_delivery =online_delivery ,
                            table_booking= table_booking ,avg_cost= avg_cost ))


# 🔹 Result page 
@app.route('/result/<int:pred>')
def result(pred):
    proba = float (request.args.get('prob',0))
    online_delivery = int(request.args.get('online_delivery',0))
    table_booking = int(request.args.get('table_booking',0))
    avg_cost = float(request.args.get('avg_cost',0))

    if pred == 1:
        message = f"✅ This restaurant is likely to be SUCCESSFUL! (confidence:{proba:.2f})"
        suggestion = generate_suggestion(
            online_delivery,
            table_booking,
            avg_cost,
            pred
        )

    else:
        message = f"❌ This restaurant may NOT be successful. (confidence:{proba:.2f})"

        suggestion = generate_suggestion(
            online_delivery,
            table_booking,
            avg_cost,
            pred
        )

    return render_template('result.html', message=message, suggestion=suggestion)

def generate_suggestion(online_delivery, table_booking, avg_cost, pred):

    suggestions = []

    # 🔹 If SUCCESS → improvement tips
    if pred == 1:
        suggestions.append("🎉 Great! Your restaurant is performing well.")

        if online_delivery == 0:
            suggestions.append("🚀 Adding online delivery can further boost your growth.")

        if table_booking == 0:
            suggestions.append("📅 Table booking can enhance customer experience.")

        if avg_cost > 1500:
            suggestions.append("💰 Consider adding mid-range options to attract more customers.")

        suggestions.append("⭐ Maintain food quality and consistency.")
        suggestions.append("📢 Keep engaging customers through offers and marketing.")

    # 🔹 If FAILURE → corrective tips
    else:
        if online_delivery == 0:
            suggestions.append("🚀 Enable online delivery to increase reach and orders.")

        if table_booking == 0:
            suggestions.append("📅 Add table booking feature to attract more customers.")

        if avg_cost > 1500:
            suggestions.append("💰 Consider reducing pricing or adding budget-friendly options.")
        elif avg_cost < 300:
            suggestions.append("📈 Improve perceived value or menu variety to increase revenue.")

        suggestions.append("⭐ Improve food quality and customer experience.")
        suggestions.append("📢 Increase marketing and customer engagement.")
        suggestions.append("🍽️ Expand menu with trending cuisines.")

    return "<br>".join(suggestions)

if __name__ == '__main__':
  import os
  port = int(os.environ.get("PORT",10000))
  app.run(host='0.0.0.0', port = port)
