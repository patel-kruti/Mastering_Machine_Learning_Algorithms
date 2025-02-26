# Step 1: Install and Load dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 2: Create dataset
# Example dataset (Area in square feet vs. Price in $1000s)
area = np.array([500, 700, 1000, 1200, 1500]).reshape(-1, 1)
price = np.array([150, 200, 250, 300, 350])  # Labels

# Step 3: Train a model
model = LinearRegression()
model.fit(area, price)  # Train model

# Step 4: Make predictions
new_area = np.array([[1100]])  # Predict price for 1100 sqft
predicted_price = model.predict(new_area)
print(f"Predicted Price: $ {predicted_price[0]*1000}")

# ----------------- Output -----------------
# Predicted Price: $ 273885.3503184713

