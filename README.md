# 🔍 Duplicate Question Pair Detection using NLP and Machine Learning

## 📖 Overview

This project is a Machine Learning and Natural Language Processing (NLP) application that predicts whether two questions are duplicates, meaning they convey the same intent even if they are phrased differently.

The model is trained on the Quora Question Pairs dataset and uses a combination of text preprocessing, feature engineering, and Bag-of-Words vectorization to classify question pairs as either **Duplicate** or **Non-Duplicate**.

An interactive Streamlit web application has been developed to allow users to test the model by entering any two questions.

---

## ✨ Features

- Predicts whether two questions are duplicates.
- Performs NLP-based text preprocessing.
- Uses handcrafted similarity features.
- Converts text into numerical vectors using Bag-of-Words.
- Interactive and easy-to-use Streamlit interface.
- Fast prediction with a trained machine learning model.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- BeautifulSoup
- FuzzyWuzzy
- Distance
- Streamlit
- Pickle

---

## 📂 Project Structure

```
Duplicate-Question-Pairs/
│
├── app.py
├── helper.py
├── model.pkl
├── cv.pkl
├── requirements.txt
├── train.csv
├── README.md
│
└── Notebooks/
    ├── initial_eda.ipynb
    ├── only-bow.ipynb
    ├── bow_with_basic_features.ipynb
    └── bow_with_preprocessing_and_advanced_features.ipynb
```

---

## 📊 Workflow

### 1. Data Collection
- Used the Quora Question Pairs dataset.

### 2. Data Preprocessing
- Converted text to lowercase.
- Removed HTML tags.
- Removed punctuation.
- Removed stopwords where required.
- Cleaned and normalized text.

### 3. Feature Engineering

The following features were extracted:

- Question length
- Character count
- Word count
- Number of common words
- Total unique words
- Word share ratio
- Token-based features
- Fuzzy matching scores
- Longest common substring
- Other similarity-based features

### 4. Text Vectorization

Bag-of-Words (CountVectorizer) was used to convert text into numerical vectors.

### 5. Model Training

The engineered features and Bag-of-Words vectors were combined and used to train a machine learning model.

The trained model was saved using Pickle for deployment.

---

## 🚀 How to Run the Project

### Clone the repository

```bash
git clone https://github.com/your-username/Duplicate-Question-Pairs.git
```

### Move to the project folder

```bash
cd Duplicate-Question-Pairs
```

### Install the required libraries

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📷 Application

The Streamlit application allows users to:

- Enter the first question.
- Enter the second question.
- Click the **Predict** button.
- View whether the questions are **Duplicate** or **Non-Duplicate**.

You can add screenshots of your application here.

---

## 📈 Future Improvements

- Use Sentence Transformers (SBERT).
- Implement BERT or RoBERTa embeddings.
- Build a deep learning model using LSTM or Transformers.
- Deploy the application on the cloud.
- Create a REST API for predictions.

---

## 📚 Dataset

**Quora Question Pairs Dataset**

The dataset contains thousands of labeled question pairs and is commonly used for semantic similarity and duplicate question detection tasks.

---

## 🎯 What I Learned

Through this project, I gained practical experience in:

- Natural Language Processing
- Text Preprocessing
- Feature Engineering
- Machine Learning
- Streamlit Deployment
- Model Serialization with Pickle
- Working with real-world NLP datasets

---

## 🤝 Contributing

Contributions are welcome.

Feel free to fork the repository, make improvements, and submit a pull request.

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

---
