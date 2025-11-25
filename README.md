# Anti-Spam Email Classifier

A machine learning-powered email spam detection system built with Python and Streamlit. This project uses natural language processing techniques and multiple classification algorithms to identify spam emails with high accuracy.

## 🚀 Features

- **Interactive Web Interface**: User-friendly Streamlit app for real-time spam detection
- **Multiple ML Models**: Comparison of Logistic Regression, SVM, Naive Bayes, and Random Forest
- **Text Preprocessing**: Advanced NLP pipeline with tokenization, stemming, and TF-IDF vectorization
- **Data Visualization**: Word clouds and distribution analysis for spam vs ham emails
- **Model Persistence**: Trained models saved for quick deployment

## 📁 Project Structure

```
Anti-Spam/
├── app/
│   └── main.py                 # Streamlit web application
├── data/
│   ├── clean/
│   │   ├── X_vectorized.npz    # Preprocessed feature vectors
│   │   └── y_labels.csv        # Target labels
│   └── raw/
│       └── DataSet_Emails.csv  # Original email dataset
├── models/
│   ├── best_model.pkl          # Trained classifier (Linear SVM)
│   └── tfidf_vectorizer.pkl    # TF-IDF vectorizer
├── notebooks/
│   ├── 01_preprocess_EDA.ipynb # Data preprocessing and analysis
│   └── 02_training_eval.ipynb  # Model training and evaluation
├── README.md
└── requirements.txt
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/9asdaoui/Anti_Spam.git
   cd Anti-Spam
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data (if not already downloaded):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   ```

## 🚀 Usage

### Web Application

Run the Streamlit app for interactive spam detection:

```bash
cd app
streamlit run main.py
```

The web interface will open in your browser where you can:
- Enter email content in the text area
- Click "Check" to classify the email as spam or ham
- Get instant results with visual indicators

### Jupyter Notebooks

Explore the data science workflow:

1. **Data Preprocessing & EDA**: `notebooks/01_preprocess_EDA.ipynb`
   - Load and clean email dataset
   - Exploratory data analysis
   - Text preprocessing pipeline
   - Word cloud generation

2. **Model Training & Evaluation**: `notebooks/02_training_eval.ipynb`
   - Train multiple classification models
   - Compare model performance
   - Save the best performing model

## 📊 Model Performance

The project evaluates four different classification algorithms:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| **Linear SVM** | **Best** | **Best** | **Best** | **Best** |
| Logistic Regression | High | High | High | High |
| Naive Bayes | Good | Good | Good | Good |
| Random Forest | Good | Good | Good | Good |

*Note: Linear SVM was selected as the best model based on overall performance metrics.*

## 🔧 Technical Details

### Text Preprocessing Pipeline

1. **Data Cleaning**: Remove null values and duplicates
2. **Text Combination**: Merge subject and body content
3. **Tokenization**: Split text into individual words
4. **Punctuation Removal**: Filter out non-alphabetic characters
5. **Stop Words Removal**: Remove common English stop words
6. **Stemming**: Reduce words to their root forms using Porter Stemmer
7. **Vectorization**: Convert text to numerical features using TF-IDF

### Features Used

- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency
- **Combined Text**: Email subject + body content
- **Preprocessed Text**: Cleaned, tokenized, and stemmed text

## 📈 Data Analysis

The project includes comprehensive exploratory data analysis:

- **Class Distribution**: Visualization of spam vs ham email distribution
- **Word Clouds**: Visual representation of most frequent words in spam and ham emails
- **Statistical Analysis**: Descriptive statistics of the dataset

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**[Your Name]** - *Initial work* - [9asdaoui](https://github.com/9asdaoui)

## 🙏 Acknowledgments

- Dataset contributors for providing the email spam dataset
- Scikit-learn community for machine learning tools
- Streamlit team for the amazing web framework
- NLTK developers for natural language processing tools

## 🔮 Future Enhancements

- [ ] Add more sophisticated feature engineering
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add email attachment analysis
- [ ] Create API endpoints for integration
- [ ] Add model retraining capabilities
- [ ] Implement real-time learning from user feedback

---

**Note**: Make sure to have the trained models (`best_model.pkl` and `tfidf_vectorizer.pkl`) in the `models/` directory before running the Streamlit app.