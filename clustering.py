import pandas as pd

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess_text

# Load dataset
df = pd.read_excel(
    'datasets/healthcare_dataset.xlsx'
)

# Preprocess text
df['Processed'] = df['Feedback'].apply(
    preprocess_text
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(
    df['Processed']
)

# KMeans clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

# Assign clusters
df['Cluster'] = kmeans.fit_predict(X)

# Save clustered dataset
df.to_excel(
    'datasets/clustered_output.xlsx',
    index=False
)

print("Clustering completed!")