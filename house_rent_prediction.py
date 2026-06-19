import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("dataset.csv")

X = df[["Location", "Area", "Bedrooms", "Bathrooms"]]
y = df["Rent"]

# Convert Location text into machine-readable values
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Location"])
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

print("House Rent Prediction System")

location = input("Enter Location: ")
area = float(input("Enter Area (sq ft): "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))

new_house = pd.DataFrame({
    "Location": [location],
    "Area": [area],
    "Bedrooms": [bedrooms],
    "Bathrooms": [bathrooms]
})

predicted_rent = model.predict(new_house)

print(f"\nPredicted Rent: ₹{predicted_rent[0]:.2f}")
