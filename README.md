# Resume Screening with Python

This project aims to screen resumes and categorize them into different job categories using natural language processing (NLP) techniques and machine learning algorithms.

## Introduction

Resume screening is an essential task for recruiters and HR professionals to efficiently filter through large volumes of resumes and identify potential candidates for job positions. This project automates the resume screening process using Python, NLP, and machine learning.

## Data

The dataset used in this project is the [Updated Resume Dataset](https://www.kaggle.com/datamunge/sign-language-mnist) from Kaggle. It contains resumes labeled with different job categories.

## Approach

1. **Data Preprocessing**: Clean the text of resumes by removing URLs, mentions, hashtags, punctuation, and stopwords.

2. **Exploratory Data Analysis (EDA)**: Visualize the distribution of job categories and common words in resumes using word clouds and frequency analysis.

3. **Feature Engineering**: Extract features from the text using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

4. **Model Training**: Train a machine learning classifier (K-Nearest Neighbors) to classify resumes into different job categories.

5. **Model Evaluation**: Evaluate the performance of the classifier using accuracy and classification reports.

## Usage

1. **Dataset**: Download the `UpdatedResumeDataSet.csv` dataset from [Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist) and place it in the project directory.

2. **Running the Code**: Execute the provided Python script `resume_screening.py` in your Python environment.

## Repository Structure

```
.
├── UpdatedResumeDataSet.csv       # Resume Dataset
├── resume_screening.py            # Python script for resume screening
└── README.md                      # This README file
```

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- nltk
- wordcloud

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
