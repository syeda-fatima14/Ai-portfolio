from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data (hours studied vs marks)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([35, 40, 50, 55, 60])

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])

print("Predicted marks for 6 hours study:", prediction[0])
