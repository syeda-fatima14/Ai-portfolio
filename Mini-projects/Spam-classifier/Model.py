from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data
emails = [
    "Win money now",
    "Meeting scheduled tomorrow",
    "Limited offer click now",
    "Project discussion"
]

labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = LogisticRegression()
model.fit(X, labels)

test_email = ["Free money offer"]
test_vector = vectorizer.transform(test_email)

prediction = model.predict(test_vector)

print("Spam" if prediction[0] == 1 else "Not Spam")
