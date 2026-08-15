# 🐦 Twitter Sentiment Analysis

A Machine Learning project that classifies Twitter posts into three sentiment categories:

- 🔴 Negative
- ⚪ Neutral
- 🟢 Positive

The project uses **TF-IDF** for text feature extraction and **Logistic Regression** for sentiment classification.

---

## 📌 About the Project

Sentiment analysis is a Natural Language Processing (NLP) technique used to determine the emotional tone of text.

In this project, a Twitter sentiment dataset is cleaned and processed before training a machine learning model to classify tweets as Negative, Neutral, or Positive.

---

## 🚀 Features

- 🧹 Text preprocessing and cleaning
- 🔤 TF-IDF feature extraction
- 🤖 Logistic Regression classifier
- 📊 Accuracy evaluation
- 📋 Classification report
- 📈 Confusion matrix
- 💬 Custom sentiment prediction

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Natural Language Processing (NLP)
- TF-IDF
- Logistic Regression
- Google Colab

---

## 📂 Dataset

The project uses the **Twitter Sentiment Dataset** containing tweets labeled as:

| Label | Sentiment |
|---|---|
| -1 | Negative |
| 0 | Neutral |
| 1 | Positive |

The dataset is used locally for training and testing and is not included in this repository.

---

## ⚙️ Machine Learning Workflow

```text
Twitter Dataset
       ↓
Data Cleaning
       ↓
Remove Missing Values
       ↓
Text Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Train/Test Split
       ↓
Logistic Regression
       ↓
Predictions
       ↓
Evaluation
       ↓
Sentiment Classification
```

## 📊 Model Performance

The Logistic Regression model achieved:

# 🎯 Test Accuracy: 85.46%

Classification Report

|Sentiment| Precision| Recall| F1-Score|
|---:|---|---|---|
|Negative| 0.86| 0.74| 0.79|
|Neutral| 0.81| 0.95| 0.87|
|Positive| 0.90| 0.84| 0.87|

The model was evaluated on 32,593 test samples.

--

## 🧠 Model Details

- TF-IDF
- TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical features that the machine learning model can understand.

The project uses:

-Maximum 10,000 features
-Unigrams and bigrams
-English stop-word removal
# Logistic Regression
- Logistic Regression is used as the classification algorithm to predict one of the three sentiment classes.

## 💬 Example

The trained model can classify custom text such as:
"I absolutely loved this movie!"

Possible prediction:
Sentiment: Positive

Another example:
"This service was terrible."

Possible prediction:
Sentiment: Negative

## 📈 Results
- The confusion matrix shows that the model performs particularly well at identifying Neutral and Positive tweets.
 
- The overall test accuracy of 85.46% demonstrates that the model can effectively classify sentiment in the provided dataset.

## 🔮 Future Improvements
- Try advanced NLP models such as BERT
- Add a web interface using Streamlit
- Support real-time tweet analysis
- Improve text preprocessing
- Compare multiple machine learning algorithms
- Deploy the model as a web application

### 👩‍💻 Author
Anushank Kumar Singh
Machine Learning / Python Project

# ⭐ If you found this project useful, consider giving the repository a star!

