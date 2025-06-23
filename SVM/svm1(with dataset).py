# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import machine learning tools from scikit-learn
from sklearn.svm import SVC                      # For Support Vector Classification
from sklearn.metrics import accuracy_score       # For evaluating model performance
from sklearn.preprocessing import LabelEncoder   # For encoding categorical labels

# Load the dataset from a CSV file
df = pd.read_csv('iris_database.csv')

# Select two features for simplicity and 2D visualization (sepal length & sepal width)
X = df.iloc[:, [0, 1]].values   # Feature matrix
Y = df.iloc[:, -1].values       # Target labels (species)

# Encode string labels (like "setosa") into integers (like 0, 1, 2)
le = LabelEncoder()
Y = le.fit_transform(Y)

# Initialize and train a Support Vector Classifier with a linear kernel
model = SVC(kernel='linear')
model.fit(X, Y)

# Predict using the trained model on the same data
y_predict = model.predict(X)

# Print the accuracy of the model on this dataset
print("Accuracy Score: ", accuracy_score(Y, y_predict))

# Visualize the data points in a 2D scatter plot
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr')  # Red/blue color map for classes

# Extract weights and bias from the trained SVM model to plot decision boundary
weight = model.coef_[0]
bias = model.intercept_[0]

# Create x-values for drawing the decision line
x_value = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)

# Calculate corresponding y-values using the SVM decision boundary formula
# Equation: w0*x + w1*y + b = 0  =>  y = -(w0*x + b)/w1
y_value = -(weight[0] * x_value + bias) / weight[1]

# Plot the decision boundary
plt.plot(x_value, y_value, '-k')  # Black line

# Add labels and title for better readability
plt.xlabel("Sepal Length (cm)")  # Previously labeled "Prices"
plt.ylabel("Sepal Width (cm)")   # Previously labeled "Houses"
plt.title("Iris Classification with Linear SVM")

# Show the plot
plt.show()
