[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Natural Language Processing : Authorship Identification

## Objectives

1. To study a wide variety of natural language processing techniques and compare the performance.
2. Tune the any model to obtain 90 % test accuracy.

## License

The source code hosted in this repository is shared under [MIT license](LICENSE).

## Sponsor

DataDisca Pty Ltd, Melbourne, Australia

[https://www.datadisca.com](https://www.datadisca.com)

## Dataset

Below mentioned 5 authors were selected as the sample authors for this project.
In order to generate dataset, multiple books from each author in plain text format is used from [Project Gutenberg](https://www.gutenberg.org/).
Each source file is covered under respective licenses by [Project Gutenberg](https://www.gutenberg.org/) and strictly used only for research purposes. Raw files are downloaded from the mirrors using a script as per instrutions in [terms of use](https://www.gutenberg.org/policy/terms_of_use.html).
 
###  Download raw files

Download the text (Plain Text UTF-8) of at least 5 books for train, test, validation splits and 4 books for the seperate validation dataset from each author in the following table.  
* Script: [Dataset_Prepare\download_dataset.ipynb](Dataset_Prepare/download_dataset.ipynb)

| Author | URL |
|--------|-----|
| Charles Dickens |  https://www.gutenberg.org/ebooks/author/37  |
| Jane Austen |  https://www.gutenberg.org/ebooks/author/68  |
| Sir Arthur Conan Doyle | https://www.gutenberg.org/ebooks/author/69  |
| George Eliot |  https://www.gutenberg.org/ebooks/author/90  |
| Jules Verne |  https://www.gutenberg.org/ebooks/author/60  |
  
###  Extract records as CSV files

Extract mutually exclusive records of length L words, from the text of each book for train, test, validation splits as "dataset.csv".  
* Script: [Dataset_Prepare\extract.ipynb](Dataset_Prepare/extract.ipynb)
  + L = 50 # length of records to be extracted
  + N = 1000 # number of records for a book
  + Extracted 25000 records in total (1000 record per book (N) * 5 authors * 5 books per author)
  + Pre-processing:
   1. Break lines (sentences) at "." (period)
   2. Replace "\r\n" with " " (space)
   3. Replace double spaces with single spaces
   4. Remove all punctuations and just keep the alphanumeric characters and spaces.
   5. Remove sentances with less than 20 char to remove gibberish.
   6. Remove first 100 sentances to remove table of contents, preface etc.
   7. Remove last 250 sentances to remove Gutenbury stuff such as license etc.
   8. Convert all text to lowercase 

Extract a seperate validation dataset from another set of books as "validation_dataset.csv".  
* Script: [Dataset_Prepare\seperate_validation.ipynb](Dataset_Prepare/seperate_validation.ipynb)
  + L = 50 # length of records to be extracted
  + N = 1000 # number of records for a book
  + Extracted 4000 records in total (200 record per book (N) * 5 authors * 4 book per author)
  + With all above pre-processing steps


## NLP Models and Techniques

### Classical Models using only "dataset.csv"

| Classifier              | Bag of Word + TF-IDF (TfidfVectorizer())         | Averaged Test Accuracy | Averaged Validation Accuracy |
|-------------------------|-----------------|-------------------|-------------------|
| Logistic Regression     | [train_LR.ipynb](Classical_Models/train_LR.ipynb)  | 89.03             | 89.02                |
| Support Vector Machines | [train_SVM.ipynb](Classical_Models/train_SVM.ipynb) |86.25              |86.65             |
| Random Forest           | [train_RFC.ipynb](Classical_Models/train_RFC.ipynb) |72.67     |72.29            |
| Naive Bayes             | [train_NB.ipynb](Classical_Models/train_NB.ipynb) |89.34         |89.92             |
| XGBoost                 | [train_XGB.ipynb](Classical_Models/train_XGB.ipynb) |78.23     |77.73          |

### Deep Learning

 Classifier Model              | Source Code         | Validation Accuracy (same books) |  Validation Accuracy (seperate books) |
|-------------------------|-----------------|-------------------|-------------------|
| GRU + Glove      |keras_glove_gru.ipynb  |82.72    | 57.55   |
| LSTM + Glove  | keras_glove_lstm.ipynb |  82.34            |59.63       |
|  Bidirectional LSTM + Glove | keras_glove_bi_lstm.ipynb  |   80.18 |     59.60 |
| GRU + Word2vec  | keras_word2vec_gru.ipynb  |    76.68   |  59.63  |
| LSTM + Word2vec   | keras_word2vec_lstm.ipynb  | 70.62             |      55.58 |          
|  Bidirectional LSTM + Word2vec| keras_word2vec_bi_lstm.ipynb |69.76  |   54.80   |     

### Transformers

| Classifier Model              | Source Code         | Validation Accuracy (same books) |  Validation Accuracy (seperate books) |
|-------------------------|-----------------|-------------------|-------------------|            
|  BERT (TensorFlow Hub) - Google Colab      |keras_bert_v4_18_colab.ipynb   | 92.76  |     71.15 |  
|  BERT (TensorFlow Hub) - Local PC    |keras_bert_v4_18_3060.ipynb   | 92.76  |     71.15 | 
