import joblib

# Load the trained model and vectorizer
model = joblib.load("models/logistic_model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

# Function to make predictions on user input
def predict_gender(name):
    name = name.strip().lower()
    if not name:
        return "Invalid input"
    name_vector = vectorizer.transform([name])
    gender_pred = model.predict(name_vector)
    return gender_pred[0].capitalize()

# Interactive inference

while True:
    user_input = input("Enter a name to predict gender (or type 'exit' to stop): ")
    if user_input.lower().strip() == 'exit':
        print("Exiting...")
        break
    gender = predict_gender(user_input)
    print(f"The predicted gender for '{user_input}' is: {gender}\n")