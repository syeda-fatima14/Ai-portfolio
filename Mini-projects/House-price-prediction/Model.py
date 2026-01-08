from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data (area in sq ft vs price)
X = np.array([[500], [800], [1000], [1200], [1500]])
y = np.array([2000000, 3000000, 4000000, 5000000, 6500000])

model = LinearRegression()
model.fit(X, y)

area = [[1100]]
predicted_price = model.predict(area)

print("Predicted price for 1100 sq ft:", predicted_price[0])
