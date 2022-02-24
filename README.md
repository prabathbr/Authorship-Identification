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

## Prerequisites

Latest tested versions are mentioned inside the brackets along with the library names for reference.

1. Python (3.9.7)
2. Jupyter Notebook (6.4.6) with IPython (7.29.0)
4. Numba (0.54.0rc1)
5. Numpy (1.22.0)
6. Pandas (1.3.4)
8. Matplotlib  (3.5.0)
9. Tensorflow (2.9.0.dev20220102) including Keras (2.9.0.dev2022010308) and Tensorboard (2.8.0a20220102)
10. Scikit-learn (1.0.1)
11. Plotly (5.4.0)
12. Pydot (1.4.2) - Dependency for [tf.keras.utils.plot_model](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model)
13. Pydotplus (2.0.2) - Dependency for [tf.keras.utils.plot_model](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model) 
14. [GraphViz](https://graphviz.org/download/) (2.50.0) - Dependency for [tf.keras.utils.plot_model](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model)

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
| GRU + Glove      |[keras_glove_gru.ipynb](Deep_Learning/keras_glove_gru.ipynb)  |82.70    | 59.45   |
| LSTM + Glove  | [keras_glove_lstm.ipynb](Deep_Learning/keras_glove_lstm.ipynb) |  80.32            |58.25       |
|  Bidirectional LSTM + Glove | [keras_glove_bi_lstm.ipynb](Deep_Learning/keras_glove_bi_lstm.ipynb)  |   78.38 |     59.23 |
| GRU + Word2vec  | [keras_word2vec_gru.ipynb](Deep_Learning/keras_word2vec_gru.ipynb) |    76.68   |  59.63  |
| LSTM + Word2vec   | [keras_word2vec_lstm.ipynb](Deep_Learning/keras_word2vec_lstm.ipynb)  | 70.62             |      55.58 |          
|  Bidirectional LSTM + Word2vec| [keras_word2vec_bi_lstm.ipynb](Deep_Learning/keras_word2vec_bi_lstm.ipynb) |69.76  |   54.80   |     

### Transformers - TensorFlow Hub

| Classifier Model              | Source Code         | Validation Accuracy (same books) |  Validation Accuracy (seperate books) |
|-------------------------|-----------------|-------------------|-------------------|            
|  BERT (AdamW / seq_length = 300)     |[keras_bert_v4_18_colab.ipynb](Transformers_TF/keras_bert_v4_18_colab.ipynb)   | 92.70  |     69.63 |  

### Transformers - HuggingFace

| Classifier Model              | Source Code         | Validation Accuracy (same books) |  Validation Accuracy (seperate books) |
|-------------------------|-----------------|-------------------|-------------------|            
|  BERT (Adam / seq_length = 100)   |[keras_hf_bert_3060.ipynb](Transformers_HF/keras_hf_bert_3060.ipynb)   | 90.40  |     67.73 | 
|  DistilBERT (Adam / seq_length = 100)   |[keras_hf_DistilBERT.ipynb](Transformers_HF/keras_hf_DistilBERT.ipynb)   | 89.42  |     65.45 | 
|  RoBERTa (Adam / seq_length = 100)   |[keras_hf_RoBERTa.ipynb](Transformers_HF/keras_hf_RoBERTa.ipynb)   | 89.48  |     67.40 | 
|  XLNet (Adam / seq_length = 100)   |[keras_hf_XLNet.ipynb](Transformers_HF/keras_hf_XLNet.ipynb)   | 88.06  |     67.88 | 

