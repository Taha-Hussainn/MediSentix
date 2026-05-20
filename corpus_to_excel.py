import pandas as pd

# Read corpus
with open(
    'datasets/raw_corpus.txt',
    'r',
    encoding='utf-8'
) as file:

    lines = file.readlines()

themes = []
feedbacks = []
labels = []

# Read line-by-line
for line in lines:

    line = line.strip()

    parts = line.split('||')

    if len(parts) == 3:

        theme = parts[0]

        sentence = parts[1]

        label = parts[2].strip()

        # Convert labels
        if label == '1':
            label = 'Positive'

        elif label == '0':
            label = 'Negative'

        else:
            label = 'Neutral'

        themes.append(theme)
        feedbacks.append(sentence)
        labels.append(label)

# Create dataframe
df = pd.DataFrame({
    'Theme': themes,
    'Feedback': feedbacks,
    'Sentiment': labels
})

# Save dataset
df.to_excel(
    'datasets/healthcare_dataset.xlsx',
    index=False
)

print('Dataset created successfully!')