import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess_text

# Load dataset
df = pd.read_excel(
    'datasets/healthcare_dataset.xlsx'
)

# Remove empty rows
df.dropna(inplace=True)

# Preprocess reviews
df['Processed'] = df['Feedback'].apply(
    preprocess_text
)

# Features
X = df['Processed']

# Labels
y = df['Sentiment']

# TF-IDF
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Save model
joblib.dump(
    model,
    'models/sentiment_model.pkl'
)

joblib.dump(
    vectorizer,
    'models/vectorizer.pkl'
)

print("Model trained successfully!")