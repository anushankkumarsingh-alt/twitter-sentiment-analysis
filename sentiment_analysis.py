# Twitter Sentiment Analysis
# NLP + TF-IDF + Logistic Regression

import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


print("🐦 Twitter Sentiment Analysis")
print("----------------------------")


# 1. Load dataset

print("\n📥 Loading dataset...")

url = (
    "https://raw.githubusercontent.com/"
    "prathameshsonpatki/"
    "Twitter-Sentiment-Analysis/master/"
    "Sentiment%20Analysis%20Dataset.csv"
)

try:
    data = pd.read_csv(url)

except Exception:
    print("❌ Could not download the dataset.")
    print("Please check your internet connection.")
    exit()


# 2. Display dataset information

print(f"Dataset shape: {data.shape}")
print("\nDataset columns:")
print(data.columns.tolist())


# 3. Identify text and sentiment columns

text_column = data.columns[0]
sentiment_column = data.columns[1]


# 4. Clean text

def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags symbol
    text = re.sub(r"#", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\n🧹 Cleaning text...")

data[text_column] = data[text_column].apply(clean_text)

# Remove empty rows
data = data[data[text_column].str.len() > 0]


# 5. Separate features and labels

X = data[text_column]
y = data[sentiment_column]


print("\n📊 Sentiment distribution:")
print(y.value_counts())


# 6. Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Convert text into numerical features using TF-IDF

print("\n🔢 Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 8. Train Logistic Regression model

print("\n🧠 Training sentiment model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


# 9. Make predictions

print("\n🔮 Making predictions...")

y_pred = model.predict(X_test_tfidf)


# 10. Evaluate model

accuracy = accuracy_score(y_test, y_pred)

print("\n📊 Model Results")
print("----------------")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))


# 11. Test custom tweets

print("\n💬 Try your own text!")

while True:

    user_text = input(
        "\nEnter a tweet/message "
        "(or type 'exit' to quit): "
    )

    if user_text.lower() == "exit":
        break

    cleaned = clean_text(user_text)

    transformed = vectorizer.transform([cleaned])

    prediction = model.predict(transformed)[0]

    print(f"🤖 Predicted Sentiment: {prediction}")


print("\n🎉 Sentiment analysis complete!")
