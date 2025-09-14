# Create a sample dataset
import pandas as pd # type: ignore
import numpy as np # type: ignore

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
num_samples = 2000
study_hours = np.random.uniform(1, 10, num_samples)
sleep_hours = np.random.uniform(4, 12, num_samples)

# Create a simple rule for passing (target variable)
# Students who study more and get adequate sleep are more likely to pass
pass_exam = (
    (study_hours > 5) & 
    (sleep_hours > 6) & 
    (study_hours + 0.5*sleep_hours > 9)
).astype(int)

# Add some noise to make it more realistic
noise = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])
pass_exam = np.where(noise == 1, 1 - pass_exam, pass_exam)

# Create DataFrame
data = pd.DataFrame({
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'pass_exam': pass_exam
})

# Save to CSV
data.to_csv('student_data.csv', index=False)

print("Sample dataset created and saved as 'student_data.csv'")
print("\nFirst 10 rows:")
print(data.head(10))

print(f"\nDataset statistics:")
print(f"Total samples: {len(data)}")
print(f"Pass rate: {data['pass_exam'].mean():.2%}")