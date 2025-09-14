import tensorflow as tf # type: ignore # For using the neural network
import pandas as pd # type: ignore # For data manipulation
import joblib  # type: ignore # For loading the scaler

print("Loading model and scaler...")
# Load the trained model from file
model = tf.keras.models.load_model('edu-predict-model.keras')

# You need to load the scaler that was used during training
# If you saved it during training, load it. Otherwise, we need to recreate it
try:
    # Try to load the scaler if you saved it
    scaler = joblib.load('scaler.save')
    print("Scaler loaded!")
except:
    print("Scaler not found. Creating a new one (this may cause issues if not fitted properly).")


def predict(study_hours: float, sleep_hours: float):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'study_hours': [study_hours],
            'sleep_hours': [sleep_hours]
        })
        
        # Scale the input data using the same scaler from training
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        probability = prediction[0][0]
        return probability, 1 if probability > 0.5 else 0
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1, -1

while True:
    try:
        study_hours = float(input("Enter study hours: "))
        sleep_hours = float(input("Enter sleep hours: "))
        probability, prediction = predict(study_hours, sleep_hours)
        if prediction != -1:
            print(f"Pass probability: {probability:.4f}")
            print(f"Prediction: {'PASS' if prediction == 1 else 'FAIL'}")
        else:
            print("Prediction failed due to an error.\n")
    except ValueError:
        print("Invalid input. Please enter numeric values for hours.\n")
    except KeyboardInterrupt:
        print("\nExiting prediction loop.")
        break