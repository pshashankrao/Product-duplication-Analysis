import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

# Step 1: Read the CSV file and compare O_Bar_Code with F_Bar_Code
file_path = 'F:/Product duplication/mechanic.csv'
df = pd.read_csv(file_path)

# Display the header
header = df.columns.tolist()
print("Header:", header)

# List all Part Names where F_Bar_Code is not empty and O_Bar_Code matches F_Bar_Code
filtered_parts = df[df['F_Bar_Code'].notna() & (df['O_Bar_Code'] == df['F_Bar_Code'])]['Part Name']
print("Filtered Part Names:", filtered_parts.tolist())

# Assuming you have a feature matrix X and target vector y
# For demonstration, let's generate dummy data
X = np.random.rand(100, 10)  # Replace with actual features
y = np.random.randint(0, 2, 100)  # Replace with actual target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Calculate accuracy for various algorithms

# Gaussian Naive Bayes
model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train)
predictions_gnb = model_gnb.predict(X_test)
accuracy_gnb = accuracy_score(y_test, predictions_gnb)
print(f'Gaussian Naive Bayes Accuracy: {accuracy_gnb}')

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred)
print(f"K-Nearest Neighbors Accuracy: {accuracy_knn}")

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred)
print(f"Support Vector Machine Accuracy: {accuracy_svm}")

# Gradient Boosting Machines
gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
accuracy_gbm = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Machines Accuracy: {accuracy_gbm}")

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy_rf}")

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy_lr}")

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy_dt}")

# Expectation-Maximization (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=2)  # Adjust n_components as needed
gmm.fit(X_train)
y_pred = gmm.predict(X_test)
accuracy_gmm = accuracy_score(y_test, y_pred)
print(f"Expectation-Maximization (Gaussian Mixture Model) Accuracy: {accuracy_gmm}")

# Convolutional Neural Network (CNN)
# Reshape data for CNN
X_cnn_train = X_train.reshape(-1, 28, 28, 1)  # Adjust dimensions as needed
X_cnn_test = X_test.reshape(-1, 28, 28, 1)  # Adjust dimensions as needed

cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Adjust input shape as needed
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_cnn_train, y_train, epochs=10)  # Adjust epochs as needed
loss, accuracy_cnn = cnn.evaluate(X_cnn_test, y_test)
print(f"Convolutional Neural Network (CNN) Accuracy: {accuracy_cnn}")

# VGG16
# Reshape data for VGG16
X_vgg16_train = np.repeat(X_train.reshape(-1, 28, 28, 1), 3, axis=-1)  # Convert to 3 channels
X_vgg16_test = np.repeat(X_test.reshape(-1, 28, 28, 1), 3, axis=-1)  # Convert to 3 channels

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Adjust input shape as needed
model = Sequential([
    vgg16,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_vgg16_train, y_train, epochs=10)  # Adjust epochs as needed
loss, accuracy_vgg16 = model.evaluate(X_vgg16_test, y_test)
print(f"VGG16 Accuracy: {accuracy_vgg16}")

# Step 3: Show inverse matrix
# Assuming A is your matrix
A = np.random.rand(3, 3)  # Replace with actual matrix
matrix = np.array([[1, 2], [3, 4]])
inverse_matrix = np.linalg.inv(matrix)
print("Inverse Matrix:")
print(inverse_matrix)

try:
    inverse_matrix = np.linalg.inv(A)
    print("Inverse Matrix:\n", inverse_matrix)
except np.linalg.LinAlgError:
    print("Matrix is singular and cannot be inverted.")

# Step 4: Calculate the total price
total_price = df['Price'].sum()
print("Total Price:", total_price)

# Step 5: Count unique from Part Name and Category and draw a graph
unique_parts = df['Part Name'].nunique()
unique_categories = df['Category'].nunique()

print(f"Unique Part Names: {unique_parts}")
print(f"Unique Categories: {unique_categories}")

# Drawing a graph
plt.bar(['Part Names', 'Categories'], [unique_parts, unique_categories])
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('Unique Count of Part Names and Categories')
plt.show()
