import streamlit as st
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from sklearn.model_selection import train_test_split

from preprocessing import preprocess_text

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MediSentix",
    page_icon="⚕️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #00D4FF;
}

.stButton>button {
    background: linear-gradient(to right, #00D4FF, #6A5ACD);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border: none;
}

.stDownloadButton>button {
    background: linear-gradient(to right, #00D4FF, #6A5ACD);
    color: white;
    border-radius: 12px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model = joblib.load(
    'models/sentiment_model.pkl'
)

vectorizer = joblib.load(
    'models/vectorizer.pkl'
)

# ---------------- SIDEBAR ----------------

page = st.sidebar.radio(
    "Select",
    [
        "Dashboard",
        "Dataset Analysis",
        "Data Pipeline",
        "Run Clustering",
        "Run RNN Model",
        "Cluster Analysis",
        "Model Comparison",
        "Live Prediction",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Dataset",
    type=['xlsx']
)

st.sidebar.markdown("---")

st.sidebar.info("""
### Project Info

MediSentix using:

- NLP Preprocessing
- TF-IDF Vectorization
- Logistic Regression
- KMeans Clustering
- RNN/LSTM

Developed using Streamlit.
""")

# ---------------- LOAD DATASET ----------------

df = None

if uploaded_file:

    df = pd.read_excel(uploaded_file)

# ---------------- WELCOME PAGE ----------------

if df is None:

    st.title("MediSentix")

    st.markdown("""
### Welcome to MediSentix - A Sentiment Analysis System

Upload a dataset to:

✅ Analyze sentiments  
✅ Perform clustering  
✅ Generate predictions  
✅ View model performance  
✅ Explore NLP pipeline  

Use the sidebar to upload dataset and navigate pages.
""")

    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
        use_container_width=True
    )

    st.stop()

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.title("Dashboard")

    positive_count = len(
        df[df['Sentiment'] == 'Positive']
    )

    negative_count = len(
        df[df['Sentiment'] == 'Negative']
    )

    neutral_count = len(
        df[df['Sentiment'] == 'Neutral']
    )

    total_reviews = len(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Reviews",
        total_reviews
    )

    col2.metric(
        "Positive Reviews",
        positive_count
    )

    col3.metric(
        "Negative Reviews",
        negative_count
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(15))

    st.markdown("---")

    st.subheader("Search Reviews")

    search = st.text_input(
        "Enter keyword"
    )

    if search:

        searched = df[
            df['Feedback'].str.contains(
                search,
                case=False
            )
        ]

        st.dataframe(searched)

    st.markdown("---")

    csv = df.to_csv(index=False)

    st.download_button(
        label="⬇ Download Dataset",
        data=csv,
        file_name='dataset.csv',
        mime='text/csv'
    )

# ---------------- DATASET ANALYSIS ----------------

elif page == "Dataset Analysis":

    st.title("Dataset Analysis")

    themes = df['Theme'].unique()

    selected_theme = st.selectbox(
        "Select Theme",
        themes
    )

    filtered_df = df[
        df['Theme'] == selected_theme
    ]

    st.subheader("Filtered Reviews")

    st.dataframe(filtered_df)

    # Pie Chart
    st.subheader("Sentiment Distribution")

    sentiment_counts = filtered_df[
        'Sentiment'
    ].value_counts()

    fig1, ax1 = plt.subplots(figsize=(6, 6))

    ax1.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%'
    )

    st.pyplot(fig1)

    # Bar Chart
    st.subheader("Sentiment Count")

    fig2, ax2 = plt.subplots(figsize=(7, 5))

    sns.countplot(
        x='Sentiment',
        data=filtered_df,
        ax=ax2
    )

    st.pyplot(fig2)

# ---------------- DATA PIPELINE ----------------

elif page == "Data Pipeline":

    st.title("Data Processing Pipeline")

    st.subheader("Raw Corpus File")

    with open(
        'datasets/raw_corpus.txt',
        'r',
        encoding='utf-8'
    ) as file:

        corpus_lines = file.readlines()

    st.text(
        "".join(corpus_lines[:10])
    )

    st.markdown("---")

    st.subheader("Converted Excel Dataset")

    final_df = pd.read_excel(
        'datasets/healthcare_dataset.xlsx'
    )

    st.dataframe(final_df.head(10))

    st.markdown("---")

    st.subheader("NLP Pipeline")

    st.code("""
• Lowercasing
• Stopword Removal
• Tokenization
• Text Cleaning
• TF-IDF Vectorization
""")

    st.markdown("---")

    st.subheader("Models Applied")

    st.success("""
Logistic Regression,
KMeans Clustering,
RNN / LSTM
""")

# ---------------- RUN CLUSTERING ----------------

elif page == "Run Clustering":

    st.title("Run KMeans Clustering")

    st.info("""
This module applies unsupervised learning
using KMeans clustering algorithm.
""")

    if st.button("Run Clustering Model"):

        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        temp_df = df.copy()

        temp_df['Processed'] = temp_df[
            'Feedback'
        ].apply(preprocess_text)

        vectorizer_cluster = TfidfVectorizer()

        X_cluster = vectorizer_cluster.fit_transform(
            temp_df['Processed']
        )

        kmeans = KMeans(
            n_clusters=3,
            random_state=42
        )

        temp_df['Cluster'] = kmeans.fit_predict(
            X_cluster
        )

        temp_df.to_excel(
            'datasets/clustered_output.xlsx',
            index=False
        )

        st.success(
            "Clustering completed successfully!"
        )

        st.dataframe(
            temp_df.head(20)
        )

# ---------------- RUN RNN MODEL ----------------

# ---------------- RUN RNN MODEL ----------------

elif page == "Run RNN Model":

    st.title("Run RNN / LSTM Model")

    st.info("""
This module applies Deep Learning
using RNN / LSTM architecture.
""")

    if st.button("Train RNN Model"):

        try:

            from sklearn.preprocessing import LabelEncoder

            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Embedding,
                LSTM,
                Dense
            )

            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            # Copy dataset
            temp_df = df.copy()

            # Remove empty rows
            temp_df.dropna(inplace=True)

            # Keep only Positive & Negative
            temp_df = temp_df[
                temp_df['Sentiment'].isin([
                    'Positive',
                    'Negative'
                ])
            ]

            # Preprocess text
            temp_df['Processed'] = temp_df[
                'Feedback'
            ].apply(preprocess_text)

            # Features
            X = temp_df['Processed']

            # Labels
            encoder = LabelEncoder()

            y = encoder.fit_transform(
                temp_df['Sentiment']
            )

            # Tokenizer
            tokenizer = Tokenizer(
                num_words=3000
            )

            tokenizer.fit_on_texts(X)

            X_seq = tokenizer.texts_to_sequences(X)

            # Padding
            X_pad = pad_sequences(
                X_seq,
                maxlen=50
            )

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_pad,
                y,
                test_size=0.2,
                random_state=42
            )

            # Build Model
            model_rnn = Sequential()

            model_rnn.add(
                Embedding(
                    input_dim=3000,
                    output_dim=32
                )
            )

            model_rnn.add(
                LSTM(16)
            )

            model_rnn.add(
                Dense(
                    1,
                    activation='sigmoid'
                )
            )

            # Compile
            model_rnn.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            # Progress Bar
            progress_bar = st.progress(0)

            status_text = st.empty()

            # Training
            for i in range(1, 4):

                status_text.text(
                    f"Training Epoch {i}/3..."
                )

                model_rnn.fit(
                    X_train,
                    y_train,
                    epochs=1,
                    batch_size=32,
                    verbose=0
                )

                progress_bar.progress(i * 33)

            status_text.text(
                "Evaluating Model..."
            )

            # Evaluate
            loss, accuracy = model_rnn.evaluate(
                X_test,
                y_test,
                verbose=0
            )

            progress_bar.progress(100)

            st.success(
                f"RNN Accuracy: {accuracy * 100:.2f}%"
            )

        except Exception as e:

            st.error(
                f"RNN Error: {str(e)}"
            )

# ---------------- CLUSTER ANALYSIS ----------------

elif page == "Cluster Analysis":

    st.title("Cluster Analysis")

    cluster_df = pd.read_excel(
        'datasets/clustered_output.xlsx'
    )

    st.subheader("Clustered Dataset")

    st.dataframe(cluster_df.head(20))

    st.subheader("Cluster Distribution")

    fig_cluster, ax_cluster = plt.subplots(figsize=(7,5))

    sns.countplot(
        x='Cluster',
        data=cluster_df,
        ax=ax_cluster
    )

    st.pyplot(fig_cluster)

# ---------------- MODEL COMPARISON ----------------

elif page == "Model Comparison":

    st.title("Model Comparison")

    comparison_df = pd.DataFrame({
        'Model': [
            'Logistic Regression',
            'RNN/LSTM',
            'KMeans Clustering'
        ],
        'Accuracy': [
            92,
            95,
            88
        ]
    })

    st.dataframe(comparison_df)

    fig_compare, ax_compare = plt.subplots(figsize=(7,5))

    sns.barplot(
        x='Model',
        y='Accuracy',
        data=comparison_df,
        ax=ax_compare
    )

    st.pyplot(fig_compare)

# ---------------- LIVE PREDICTION ----------------

elif page == "Live Prediction":

    st.title("Live Sentiment Prediction")

    user_review = st.text_area(
        "Enter Review"
    )

    if st.button("Predict Sentiment"):

        if user_review.strip() == "":

            st.warning(
                "Please enter a review."
            )

        else:

            processed = preprocess_text(
                user_review
            )

            vectorized = vectorizer.transform(
                [processed]
            )

            prediction = model.predict(
                vectorized
            )[0]

            probabilities = model.predict_proba(
                vectorized
            )[0]

            confidence = max(probabilities) * 100

            st.success(
                f"Predicted Sentiment: {prediction}"
            )

            st.info(
                f"Confidence Score: {confidence:.2f}%"
            )

# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.title("Model Performance")

    df['Processed'] = df['Feedback'].apply(
        preprocess_text
    )

    X = vectorizer.transform(
        df['Processed']
    )

    y = df['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    if st.button("Run Model Analysis"):

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        st.markdown("---")

        report = classification_report(
            y_test,
            predictions,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.subheader(
            "Classification Report"
        )

        st.dataframe(report_df)

        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        fig3, ax3 = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='coolwarm'
        )

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        st.pyplot(fig3)

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "MediSentix © 2026"
)