# Twitter Sentiment Analysis with BERT

A machine learning project that performs sentiment analysis on Twitter data using BERT (Bidirectional Encoder Representations from Transformers). The model classifies tweets into four categories: Positive, Negative, Neutral, and Irrelevant.

## 📊 Project Overview

This project implements a fine-tuned BERT model for multi-class sentiment classification, achieving **96% accuracy** on the validation dataset. The model processes Twitter text data and predicts sentiment with high precision across all categories.

## 🎯 Features

- **Multi-class Classification**: Supports 4 sentiment categories (Positive, Negative, Neutral, Irrelevant)
- **BERT-based Model**: Uses pre-trained BERT-base-uncased for robust text understanding
- **Text Preprocessing**: Comprehensive cleaning including URL removal, mention handling, and lemmatization
- **High Performance**: Achieves 96% accuracy with balanced precision and recall
- **Easy Prediction Interface**: Simple function for real-time sentiment prediction

## 📋 Requirements

```bash
transformers>=4.54.1
torch
pandas
nltk
scikit-learn
accelerate
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd sentiment-analysis-bert
```

2. Install dependencies:
```bash
pip install -U transformers accelerate torch pandas nltk scikit-learn
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

## 📁 Dataset Structure

The project expects CSV files with the following structure:
- Column 0: ID
- Column 1: Entity (e.g., Twitter, Facebook, etc.)
- Column 2: Sentiment (Positive/Negative/Neutral/Irrelevant)
- Column 3: Text content

Example:
```
2401,Borderlands,Positive,"im getting on borderlands and i will murder you..."
3364,Facebook,Irrelevant,"I mentioned on Facebook that I was struggling..."
```

## 🔧 Usage

### Training the Model

```python
# Load and preprocess data
train_df = pd.read_csv('twitter_training.csv')
val_df = pd.read_csv('twitter_validation.csv')

# Set column names
train_df.columns = ['id', 'entity', 'sentiment', 'text']
val_df.columns = ['id', 'entity', 'sentiment', 'text']

# Clean and tokenize
train_df['clean_text'] = train_df['text'].apply(clean_text)
val_df['clean_text'] = val_df['text'].apply(clean_text)

# Train the model
trainer.train()
```

### Making Predictions

```python
# Predict sentiment for new text
result = predict_sentiment("I absolutely love this product!")
print(result)  # Output: positive

result = predict_sentiment("This is the worst ever.")
print(result)  # Output: negative
```

## 🧹 Text Preprocessing

The preprocessing pipeline includes:

1. **URL Removal**: Strips out HTTP links
2. **Mention Cleaning**: Removes @username mentions
3. **Punctuation Removal**: Keeps only alphabetic characters
4. **Lowercasing**: Converts to lowercase
5. **Tokenization**: Uses NLTK word tokenization
6. **Stop Word Removal**: Removes common English stop words
7. **Lemmatization**: Reduces words to their root form

## 📈 Model Performance

| Metric | Positive | Negative | Neutral | Irrelevant | Overall |
|--------|----------|----------|---------|------------|---------|
| Precision | 0.95 | 0.97 | 0.96 | 0.95 | 0.96 |
| Recall | 0.96 | 0.98 | 0.95 | 0.94 | 0.96 |
| F1-Score | 0.96 | 0.97 | 0.96 | 0.95 | 0.96 |
| Support | 277 | 266 | 285 | 172 | 1000 |

**Overall Accuracy: 96%**

## 🔄 Training Configuration

- **Model**: BERT-base-uncased
- **Epochs**: 2
- **Batch Size**: 16 (training), 64 (validation)
- **Learning Rate**: Default AdamW optimizer
- **Evaluation Strategy**: Per epoch
- **Max Sequence Length**: 512 tokens

## 📊 Training Results

- **Final Training Loss**: 0.579
- **Final Validation Loss**: 0.152
- **Training Time**: ~71 minutes
- **Training Samples/Second**: 33.48

## 🔍 Example Predictions

```python
predict_sentiment("I absolutely love this product!")     # → positive
predict_sentiment("This is the worst ever.")             # → negative
predict_sentiment("I am coming to the borders...")       # → positive
predict_sentiment("i am sad")                            # → negative
predict_sentiment("i am angry")                          # → negative
predict_sentiment("nice shirt")                          # → positive
predict_sentiment("bbsusuwu")                           # → irrelevant
```

## 🛠️ Key Functions

### `clean_text(text)`
Preprocesses raw text for model input

### `predict_sentiment(text)`
Takes raw text input and returns predicted sentiment category

### `TwitterDataset`
PyTorch Dataset class for handling tokenized data

## 📝 Notes

- The model uses Weights & Biases (wandb) for experiment tracking
- BERT tokenizer handles text truncation and padding automatically
- Model checkpoints are saved in `./results` directory
- Training logs are stored in `./logs` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers library
- BERT paper authors (Devlin et al.)
- Twitter dataset providers
- NLTK development team

---

**Built with ❤️ using BERT and PyTorch**
