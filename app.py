from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import joblib
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import logging
import pickle
import pandas as pd

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the saved model
# Load models and feature columns



# Load Stroke Prediction Model
try:
    model = joblib.load("best_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model or feature columns: {e}")
    exit(1)

# Load Heart Attack Prediction Model
try:
    heart_model = joblib.load("heart_attack_best_model.pkl")
    heart_feature_columns = joblib.load("heart_attack_feature_columns.pkl")
except Exception as e:
    print(f"Error loading heart attack model or feature columns: {e}")
    exit(1)

# Database setup
def init_db():
    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)

init_db()

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = generate_password_hash(request.form["password"].strip())

        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                flash("Registration successful. Please login.")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Username already exists")
    return render_template("register.html")

@app.route("/home")
def home():
    if "user_id" not in session:
        flash("You must be logged in to access this page.")
        return redirect(url_for("login"))

    username = session.get("username", "Guest")
    return render_template("home.html", username=username)

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            # Mapping categorical values
            gender_map = {"Male": 0, "Female": 1, "Other": 2}
            ever_married_map = {"Yes": 1, "No": 0}
            work_type_map = {"Private": 0, "Self-employed": 1, "Government": 2, "Never_worked": 3, "Children": 4}
            residence_type_map = {"Urban": 0, "Rural": 1}
            smoking_status_map = {"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3}

            # Extract form data
            user_input = {
                "age": float(request.form["age"]),
                "hypertension": int(request.form["hypertension"]),
                "heart_disease": int(request.form["heart_disease"]),
                "avg_glucose_level": float(request.form["avg_glucose_level"]),
                "bmi": float(request.form["bmi"]),
                "gender_Female": 1 if request.form["gender"] == "Female" else 0,
                "gender_Male": 1 if request.form["gender"] == "Male" else 0,
                "gender_Other": 1 if request.form["gender"] == "Other" else 0,  # Ensure it's included
                "ever_married_Yes": 1 if request.form["ever_married"] == "Yes" else 0,
                "work_type_Govt_job": 1 if request.form["work_type"] == "Government" else 0,
                "work_type_Never_worked": 1 if request.form["work_type"] == "Never_worked" else 0,
                "work_type_Private": 1 if request.form["work_type"] == "Private" else 0,
                "work_type_Self-employed": 1 if request.form["work_type"] == "Self-employed" else 0,
                "work_type_children": 1 if request.form["work_type"] == "Children" else 0,  # Ensure it's included
                "Residence_type_Urban": 1 if request.form["Residence_type"] == "Urban" else 0,
                "smoking_status_formerly smoked": 1 if request.form["smoking_status"] == "formerly smoked" else 0,
                "smoking_status_never smoked": 1 if request.form["smoking_status"] == "never smoked" else 0,
                "smoking_status_smokes": 1 if request.form["smoking_status"] == "smokes" else 0
            }

            # Ensure missing features are added
            for col in feature_columns:
                if col not in user_input:
                    user_input[col] = 0  # Set missing feature to 0

            # Convert to DataFrame with correct order
            input_df = pd.DataFrame([user_input])[feature_columns]

            # Apply feature scaling
            numeric_features = ["age", "avg_glucose_level", "bmi"]
            input_df[numeric_features] = scaler.transform(input_df[numeric_features])

            # Log feature vector
            app.logger.info(f"Feature vector: {input_df.values.tolist()}")

            # Make prediction
            prediction_prob = model.predict_proba(input_df)[0][1]
            threshold = 0.3
            final_prediction = "Positive" if prediction_prob > threshold else "Negetive"

            # Log prediction result
            app.logger.info(f"Prediction: {final_prediction}, Probability: {prediction_prob}")

            return render_template("result.html", prediction=final_prediction, probability=prediction_prob)

        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            flash(f"Error during prediction: {e}")
            return redirect(url_for("home"))

    return render_template("predict.html")





@app.route("/heartpredict", methods=["POST", "GET"])
def heartpredict():
    try:
        if request.method == "POST":
            # Load Model & Features
            with open("heart_attack_best_model.pkl", "rb") as file:
                loaded_model = pickle.load(file)
            with open("heart_attack_feature_columns.pkl", "rb") as f:
                feature_columns = pickle.load(f)

            # User Input (Only 14 Attributes, No "education" or "TenYearCHD")
            user_input = {
                "male": int(request.form.get("male", 0)),  # Default: Female (0)
                "age": float(request.form.get("age", 0)),
                "currentSmoker": int(request.form.get("currentSmoker", 0)), 
                "cigsPerDay": float(request.form.get("cigsPerDay", 0)), 
                "BPMeds": int(request.form.get("BPMeds", 0)), 
                "prevalentStroke": int(request.form.get("prevalentStroke", 0)), 
                "prevalentHyp": int(request.form.get("prevalentHyp", 0)), 
                "diabetes": int(request.form.get("diabetes", 0)), 
                "totChol": float(request.form.get("totChol", 180)),  # Normal Cholesterol Default
                "sysBP": float(request.form.get("sysBP", 120)),  # Normal BP Default
                "diaBP": float(request.form.get("diaBP", 80)),  # Normal BP Default
                "BMI": float(request.form.get("BMI", 25)),  # Healthy BMI Default
                "heartRate": float(request.form.get("heartRate", 75)),  # Normal Heart Rate Default
                "glucose": float(request.form.get("glucose", 90)),  # Normal Glucose Default
            }

            # Ensure input matches feature names (14 attributes)
            user_input_df = pd.DataFrame([user_input], columns=feature_columns)

            # Debugging: Print user input before prediction
            print("\nUser Input DataFrame:\n", user_input_df)

            # Apply preprocessing pipeline
            processed_input = loaded_model[:-1].transform(user_input_df)

            # Predict
            prediction = loaded_model[-1].predict(processed_input)[0]
            prediction_proba = loaded_model[-1].predict_proba(processed_input)[0][1]
            result = "Positive" if prediction == 1 else "Negetive"

            return render_template("resultattack.html", prediction=result, probability=f"{prediction_proba:.2%}")

        return render_template("heartpredict.html")

    except Exception as e:
        flash(f"Unexpected error: {e}")
        return redirect(url_for("heartpredict"))

    



@app.route("/profile")
def profile():
    if "user_id" not in session:
        flash("You must be logged in to access this page.")
        return redirect(url_for("login"))
    return render_template("profile.html")

@app.route("/health-tips")
def health_tips():
    if "user_id" not in session:
        flash("You must be logged in to access this page.")
        return redirect(url_for("login"))
    return render_template("health_tips.html")

@app.route("/feedback")
def feedback():
    if "user_id" not in session:
        flash("You must be logged in to access this page.")
        return redirect(url_for("login"))
    return render_template("feedback.html")

@app.route("/about")
def about():
    if "user_id" not in session:
        flash("You must be logged in to access this page.")
        return redirect(url_for("login"))
    return render_template("about.html")
@app.route("/bmical", methods=["GET", "POST"])
def bmical():
    if "user_id" not in session:
        flash("You must be logged in to access this page.")
        return redirect(url_for("login"))
    
    bmi_result = None
    message = None
    normal_range = "18.5 - 24.9"
    
    if request.method == "POST":
        try:
            weight = float(request.form["weight"])
            height_cm = float(request.form["height"])  # height in cm
            height_m = height_cm / 100  # convert height from cm to meters
            
            if weight <= 0 or height_m <= 0:
                flash("Weight and height must be positive numbers.")
            elif height_cm < 50 or height_cm > 300:  # height validation (in cm)
                flash("Height must be between 50 cm and 300 cm.")
            else:
                bmi = weight / (height_m ** 2)  # BMI formula with height in meters
                if bmi < 18.5:
                    message = "Your BMI indicates that you are underweight. Normal BMI range: " + normal_range
                elif 18.5 <= bmi <= 24.9:
                    message = "Your BMI is within the normal range. Keep maintaining a healthy lifestyle!"
                elif bmi >= 25:
                    message = f"Your BMI is {bmi:.2f}. This indicates that you are overweight or obese. Normal BMI range: " + normal_range
                bmi_result = f"Your BMI is {bmi:.2f}."
        except ValueError:
            flash("Please enter valid numerical values for weight and height.")
    
    return render_template("bmical.html", bmi_result=bmi_result, message=message)





@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
