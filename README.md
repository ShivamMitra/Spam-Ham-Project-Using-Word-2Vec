## 🚀 Overview

This project demonstrates how to build a spam classifier by:

1.Preprocessing message text (tokenize, lowercase, remove stopwords/non-English, lemmatize).

2.Training or loading Word2Vec embeddings (via Gensim).

3.Aggregating word vectors into sentence-level vectors.

4.Applying a classifier (e.g. Naive Bayes, SVM, Random Forest) on those sentence embeddings.

5.Evaluating the model performance.

## 📁 Repository Structure
```
├── data/                 # (Optional) Raw & preprocessed datasets
├── notebooks/           # Jupyter notebooks for experiments
├── src/
│   ├── preprocess.py     # Text cleaning & tokenization
│   ├── train_embeddings.py # Train/load Word2Vec
│   ├── vectorize.py       # Sentence-level aggregation
│   ├── train_model.py     # Train classifier
│   └── evaluate.py        # Evaluate & report metrics
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── LICENSE               # Project license
```

## ⚙️ Installation & Setup
Clone the repo
```
git clone https://github.com/ShivamMitra/Spam-Ham-Project-Using-Word-2Vec.git
cd Spam-Ham-Project-Using-Word-2Vec
```
Create Python virtual environment
```
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
Install dependencies
```
pip install -r requirements.txt
```
Download NLTK data (if used)
```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 💡 Usage
1. Prepare your data
Place raw text in data/raw/

Use preprocess.py to tokenize, clean, and save results to data/processed/

Example:
```
python src/preprocess.py \
  --input data/raw/messages.csv \
  --output data/processed/messages_cleaned.csv
```
2. Train or load embeddings
Train a fresh Word2Vec model:
```
python src/train_embeddings.py \
  --input data/processed/messages_cleaned.csv \
  --model-path models/word2vec.model
Or load pretrained embeddings:

python src/train_embeddings.py \
  --load-model models/word2vec.model
```
3. Create sentence-level vectors
```   
python src/vectorize.py \
  --model models/word2vec.model \
  --input data/processed/messages_cleaned.csv \
  --output data/features/message_vectors.npz
```
5. Train classification model
```
python src/train_model.py \
  --input data/features/message_vectors.npz \
  --algo svm \
  --model-out models/spam_classifier.pkl
```
Supports algorithms like: naive_bayes, svm, random_forest.

7. Evaluate performance
```
python src/evaluate.py \
  --model models/spam_classifier.pkl \
  --vectors data/features/message_vectors.npz
```
Expected metrics: accuracy, precision, recall, F1-score, confusion matrix.

