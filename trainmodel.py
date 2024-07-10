import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the sample data
sample_data = pd.read_csv('/Users/kurubarantss/Downloads/convertedsample.csv')

# Prepare the data
X = sample_data['Description'].astype(str)
y = sample_data['Sub Category']

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_vectorized, y)

# Save the model and vectorizer
joblib.dump(model, 'sub_category_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
