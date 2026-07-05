# Spam-Ham Classification Using Word2Vec

A concise spam-ham classification project using word embeddings. This repository walks through preprocessing message text (tokenization, cleaning, lemmatization), building or loading a Word2Vec model via Gensim, converting word embeddings into sentence-level vectors, and applying classifiers (Naive Bayes, SVM, Random Forest) to detect spam vs. ham messages.

## 🚀 Overview

The project follows these steps:

1. **Preprocess** message text — tokenize, lowercase, remove stopwords/non-English words, and lemmatize.
2. **Train or load** Word2Vec embeddings using Gensim.
3. **Aggregate** word vectors into sentence-level vectors (Average Word2Vec).
4. **Train** a classifier (e.g., Naive Bayes, SVM, Random Forest) on the sentence embeddings.
5. **Evaluate** model performance using standard classification metrics.

## 📁 Repository Structure

```
├── Spam Ham Projects Using Word2vec,AvgWord2vec.ipynb   # Main notebook with full pipeline
├── README.md                                            # This file
└── LICENSE                                              # MIT License
```

## ⚙️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/ShivamMitra/Spam-Ham-Project-Using-Word-2Vec.git
cd Spam-Ham-Project-Using-Word-2Vec
```

Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn gensim nltk jupyter
```

Download the necessary NLTK data:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 💡 Usage

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open `Spam Ham Projects Using Word2vec,AvgWord2vec.ipynb`.

3. Run the cells in order to:
   - Load and clean the raw SMS/message dataset
   - Tokenize and lemmatize the text
   - Train a Word2Vec model on the corpus (or load pretrained vectors)
   - Compute the average Word2Vec vector for each message
   - Train and evaluate classifiers (Naive Bayes, SVM, Random Forest) on the resulting features

## 📊 Evaluation

The notebook reports standard classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## 🛠️ Tech Stack

- Python
- Gensim (Word2Vec)
- NLTK (text preprocessing)
- scikit-learn (classifiers & metrics)
- Jupyter Notebook

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.
