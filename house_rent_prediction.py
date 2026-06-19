import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("dataset.csv")

# Features and target
X = df[['Area', 'Bedrooms', 'Bathrooms']]
y = df['Rent']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# User Input
print("\nEnter House Details")
area = float(input("Area (sq ft): "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))

prediction = model.predict([[area, bedrooms, bathrooms]])

print(f"\nPredicted Rent: ₹{prediction[0]:.2f}")
