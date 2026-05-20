import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import preprocess_text

# Load dataset
df = pd.read_excel(
    'datasets/healthcare_dataset.xlsx'
)

# Remove empty rows
df.dropna(inplace=True)

# Convert labels
df['Sentiment'] = df['Sentiment'].replace({
    1: 'Positive',
    0: 'Negative'
})

# Preprocess text
df['Processed'] = df['Feedback'].apply(
    preprocess_text
)

# Features
X = df['Processed']

# Labels
encoder = LabelEncoder()

y = encoder.fit_transform(
    df['Sentiment']
)

# Tokenization
tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)

# Padding
X_pad = pad_sequences(
    X_seq,
    maxlen=100
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y,
    test_size=0.2,
    random_state=42
)

# Build model
model = Sequential()

model.add(
    Embedding(
        input_dim=5000,
        output_dim=128
    )
)

model.add(
    LSTM(64)
)

model.add(
    Dense(2, activation='softmax')
)

# Compile
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32
)

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f'RNN Accuracy: {accuracy * 100:.2f}%')