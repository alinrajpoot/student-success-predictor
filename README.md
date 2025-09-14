# Student Success Predictor
A complete, beginner-friendly TensorFlow implementation for predicting student academic performance. This project demonstrates the entire machine learning workflow from dataset creation to model deployment.


![](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?logo=tensorflow)
![](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![](https://img.shields.io/badge/License-MIT-green)

### 📊 Project Overview
This educational project provides a hands-on introduction to machine learning with TensorFlow. It includes:

- Synthetic dataset generation for student performance prediction
- Complete model training pipeline with extensive comments
- Interactive prediction system
- Visualizations of data and training progress
- Beginner-friendly code with detailed explanations

### 🎯 What You'll Learn
- Creating synthetic datasets for ML projects
- Preprocessing and scaling data for neural networks
- Building and training TensorFlow models
- Evaluating model performance
- Making predictions with trained models
- Handling common ML pitfalls (like data scaling issues)

### 📁 Project Structure
```txt
FirstSteps-TensorFlow/
│
├── dataset.py          # Creates sample student performance dataset
├── train.py            # Trains the TensorFlow model
├── predict.py          # Makes predictions using the trained model
├── student_data.csv    # Generated dataset (created by dataset.py)
├── my_first_model.keras # Trained model (created by train.py)
├── scaler.save         # Data scaler (created by train.py)
├── data_visualization.png  # Visualization of the dataset
├── training_history.png    # Training progress visualization
└── README.md           # This file
```

### 🚀 Getting Started

#### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

#### Installation
1. Clone the repository:
```bash
git clone https://github.com/alinrajpoot/student-success-predictor.git
cd student-success-predictor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### 📊 Usage
#### 1. Generate Sample Dataset
```bash
python dataset.py
```
This creates `student_data.csv` with synthetic student data and visualizes it.

![](https://raw.githubusercontent.com/alinrajpoot/student-success-predictor/424d599f413f55de0c307a184a6de3e931ac9440/data_visualization.png)


#### 2. Train the Model
```bash
python train.py
```
This trains the neural network, saves the model, and shows training progress.

![](https://raw.githubusercontent.com/alinrajpoot/student-success-predictor/424d599f413f55de0c307a184a6de3e931ac9440/training_history.png)


#### 3. Make Predictions
```bash
python predict.py
```
Interactively predict whether a student will pass based on their study and sleep habits.

Example output:

```txt
Enter study hours: 3
Enter sleep hours: 5
Pass probability: 0.1243
Prediction: FAIL

Enter study hours: 8
Enter sleep hours: 7
Pass probability: 0.9562
Prediction: PASS
```


### 🧠 How It Works
The model uses a simple neural network with:
- Input layer: 2 neurons (study hours, sleep hours)
- Hidden layers: 16 and 8 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation for binary classification

The synthetic dataset is generated with a logical pattern:
- Students who study more than 5 hours
- AND sleep more than 6 hours
- AND have a combined score (study + 0.5*sleep) > 9
- Are more likely to pass (with some noise added for realism)

<img src="https://i.postimg.cc/1mWSGKJk/Screenshot-from-2025-09-15-03-05-04.png" />

### 🤝 Contributing
Contributions are welcome! This is an educational project, so we particularly appreciate:

- Improved documentation
- Additional examples
- More detailed comments for beginners
- Alternative implementations

### 🙏 Acknowledgments
- TensorFlow team for excellent documentation
- Scikit-learn for data preprocessing tools
- The open-source community for invaluable learning resources
