# Import necessary libraries
import pandas as pd  # type: ignore # For data manipulation and reading CSV files
import tensorflow as tf  # type: ignore # Main deep learning library
from sklearn.model_selection import train_test_split  # type: ignore # For splitting data into train/test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore # For data preprocessing
import numpy as np  # type: ignore # For numerical operations
import matplotlib.pyplot as plt  # type: ignore # For data visualization
import joblib # type: ignore # To save the scaler

# Name of the model file to save
model_name = "edu-predict-model"

# Set random seeds for reproducibility
tf.random.set_seed(42)

# 1. LOAD AND EXPLORE THE DATA
# =============================
print("Step 1: Loading and exploring the data...")

# Read the CSV file into a pandas DataFrame
# Replace 'student_data.csv' with your actual file path
data = pd.read_csv('student_data.csv')

# Shape of the dataset
print(f"Dataset shape: {data.shape}")

# Display the first few rows to understand the data structure
print("Data preview:")
print(data.head())

# Check basic information about the dataset
print("\nData info:")
print(data.info())

# Statistical summary
print("\nStatistical summary:")
print(data.describe())

print(f"\nPass rate: {data['pass_exam'].mean():.2%}")


# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(data['study_hours'], data['sleep_hours'], 
            c=data['pass_exam'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Study Hours')
plt.ylabel('Sleep Hours')
plt.title('Study vs Sleep Hours (Red=Pass, Blue=Fail)')
plt.colorbar(label='Pass Exam (1=Yes, 0=No)')
plt.grid(True, alpha=0.3)
plt.savefig('data_visualization.png')
plt.show()


# 2. PREPARE THE DATA FOR TRAINING
# ================================
print("\nStep 2: Preparing the data for training...")

# Separate features (input variables) and target (what we want to predict)
# Assuming the last column is the target variable
# Adjust these indices based on your actual data structure
# X = data.iloc[:, :-1]  # All rows, all columns except the last one
# y = data.iloc[:, -1]   # All rows, only the last column
X = data[['study_hours', 'sleep_hours']]  # Features
y = data['pass_exam']  # Target

# If your target is categorical (like labels), we need to encode it as numbers
# if y.dtype == 'object':  # Check if target is text-based
#     le = LabelEncoder()
#     y = le.fit_transform(y)  # Convert text labels to numbers
#     print(f"Encoded classes: {le.classes_}")  # Show mapping of labels to numbers

# Split the data into training and testing sets
# train_test_split randomly divides the data so we can evaluate performance
# test_size=0.2 means 20% of data is used for testing, 80% for training
# random_state ensures the split is reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Standardize the features (mean=0, std=1)
# Neural networks perform better with normalized data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit to training data and transform
X_test_scaled = scaler.transform(X_test)  # Transform test data using training parameters

# 3. BUILD THE NEURAL NETWORK MODEL
# ==================================
print("\nStep 3: Building the neural network model...")

# Create a Sequential model (linear stack of layers)
model = tf.keras.Sequential([
    # First hidden layer with 16 neurons
    # input_shape defines the number of input features (must match your data)
    # ReLU activation helps with nonlinear problems
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    
    # Dropout layer randomly sets 20% of inputs to 0 to prevent overfitting
    # Overfitting is when the model memorizes training data but doesn't generalize well
    # tf.keras.layers.Dropout(0.2),

    # Second hidden layer with 8 neurons (half of the first layer)
    tf.keras.layers.Dense(8, activation='relu'),
    
    # Output layer - structure depends on your problem:
    # For binary classification: 1 neuron with sigmoid activation
    # For multi-class classification: neurons = number of classes with softmax activation
    # For regression: 1 neuron with linear activation (no activation function)
    tf.keras.layers.Dense(1, activation='sigmoid')  # Change this based on your problem
])

# 4. COMPILE THE MODEL
# ====================
print("\nStep 4: Compiling the model...")

# Compile the model with appropriate settings:
# - Optimizer: algorithm to adjust weights (adam is generally a good choice)
# - Loss: function to measure model error (depends on problem type)
# - Metrics: how to evaluate performance

# For binary classification:
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# For multi-class classification:
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',  # Use if labels are integers
#     metrics=['accuracy']
# )

# For regression:
# model.compile(
#     optimizer='adam',
#     loss='mean_squared_error',  # Measures average squared difference
#     metrics=['mae']  # Mean Absolute Error
# )

# Display model architecture
print("Model architecture:")
model.summary()

# 5. TRAIN THE MODEL
# ==================
print("\nStep 5: Training the model...")

# Train the model on our data
# epochs: number of passes through the entire training dataset
# batch_size: number of samples processed before updating weights
# validation_data: evaluate performance on test set after each epoch
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,  # Adjust based on your needs - more epochs = longer training
    batch_size=32,  # Typically 16, 32, 64, or 128
    verbose=1  # Show progress bar
)

# 6. EVALUATE THE MODEL
# =====================
print("\nStep 6: Evaluating the model...")

# Evaluate final performance on test data
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
# print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# 7. MAKE PREDICTIONS
# ===================
print("\nStep 7: Making predictions...")


# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Use the trained model to make predictions on new data
predictions = model.predict(X_test_scaled)
predicted_classes = (predictions > 0.5).astype("int32").flatten()


# For classification, convert probabilities to class labels
# (adjust based on your problem type)
if len(predictions[0]) == 1:  # Binary classification
    predicted_classes = (predictions > 0.5).astype("int32").flatten()
    print("First 10 predictions (class 0 or 1):")
    print(predicted_classes[:10])
    print("Corresponding actual values:")
    print(y_test[:10])
# else:  # Multi-class classification
#     predicted_classes = np.argmax(predictions, axis=1)
#     print("First 10 predictions (class indices):")
#     print(predicted_classes[:10])


# Create a DataFrame to compare actual vs predicted
results = pd.DataFrame({
    'Study_Hours': X_test['study_hours'],
    'Sleep_Hours': X_test['sleep_hours'],
    'Actual': y_test.values,
    'Predicted': predicted_classes,
    'Prediction_Probability': predictions.flatten()
})

print("\nSample predictions:")
print(results.head(10))


# Calculate accuracy of predictions
accuracy = np.mean(results['Actual'] == results['Predicted'])
print(f"\nManual accuracy check: {accuracy:.4f}")

# 8. SAVE THE MODEL FOR FUTURE USE (OPTIONAL)
# ============================================
print("\nStep 8: Saving the model...")

# Save the entire model to a file
model.save(f'{model_name}.keras')  # or use .h5 format
print(f"Model saved as '{model_name}.keras'")
print(f"You can load it later with: model = tf.keras.models.load_model('{model_name}.keras')")

# Save the scaler for future data preprocessing
joblib.dump(scaler, 'scaler.save')
print("Scaler saved as 'scaler.save'")
print("You can load it later with: scaler = joblib.load('scaler.save')")


# 9. TIPS FOR IMPROVEMENT (ADVANCED)
# ==================================
print("\nStep 9: Tips for improvement...")
print("- Try different architectures (more/fewer layers, neurons)")
print("- Tune hyperparameters (learning rate, batch size, epochs)")
print("- Use cross-validation for better performance estimation")
print("- Add more data or use data augmentation")
print("- Try regularization techniques to reduce overfitting")


#* 10. EXAMPLE: MAKE PREDICTION ON NEW DATA
print("\nExample prediction on new data:")

# Create new student data
new_student = pd.DataFrame({
    'study_hours': [7.5],
    'sleep_hours': [8.2]
})

# Scale the new data using the same scaler
new_student_scaled = scaler.transform(new_student)

# Make prediction
prediction = model.predict(new_student_scaled)
probability = prediction[0][0]
predicted_class = 1 if probability > 0.5 else 0

print(f"New student: {7.5} study hours, {8.2} sleep hours")
print(f"Pass probability: {probability:.4f}")
print(f"Prediction: {'PASS' if predicted_class == 1 else 'FAIL'}")

print("\nTraining completed! 🎉")