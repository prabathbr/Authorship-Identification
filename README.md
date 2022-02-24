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

## Datasets

Below mentioned 5 authors were selected as the sample authors for this project.
In order to generate dataset, multiple books from each author in plain text format is used from [Project Gutenberg](https://www.gutenberg.org/).
Each source file is covered under respective licenses by [Project Gutenberg](https://www.gutenberg.org/) and strictly used only for research purposes. Raw files are downloaded using the mirrors using a script as per instrutions in [terms of use](ttps://www.gutenberg.org/policy/terms_of_use.html)
 
###  Download raw files

Download the text (Plain Text UTF-8) of at least 5 books for train, test, validation splits and 4 books for the seperate validation dataset from each author in the following table.  
Script: Dataset_Prepare\download_dataset.ipynb

| Author | URL |
|--------|-----|
| Charles Dickens |  https://www.gutenberg.org/ebooks/author/37  |
| Jane Austen |  https://www.gutenberg.org/ebooks/author/68  |
| Sir Arthur Conan Doyle | https://www.gutenberg.org/ebooks/author/69  |
| George Eliot |  https://www.gutenberg.org/ebooks/author/90  |
| Jules Verne |  https://www.gutenberg.org/ebooks/author/60  |
  
###  Extract records as CSV files

Extract mutually exclusive records of length L words, from the text of each book for train, test, validation splits as "dataset.csv".  
Script: Dataset_Prepare\extract.ipynb

* L = 50 # length of records to be extracted
* N = 1000 # number of records for a book
* Extracted 25000 records in total (1000 record per book (N) * 5 authors * 5 books per author)
* Pre-processing:
  1. Break lines (sentences) at "." (period)
  2. Replace "\r\n" with " " (space)
  3. Replace double spaces with single spaces
  4. Remove all punctuations and just keep the alphanumeric characters and spaces.
  5. Remove sentances with less than 20 char to remove gibberish.
  6. Remove first 100 sentances to remove table of contents, preface etc.
  7. Remove last 250 sentances to remove Gutenbury stuff such as license etc.
  8. Convert all text to lowercase 

Extract a seperate validation dataset from another set of books as "validation_dataset.csv".  
Script: Dataset_Prepare\seperate_validation.ipynb
  + L = 50 # length of records to be extracted
  + N = 1000 # number of records for a book
  + Extracted 4000 records in total (200 record per book (N) * 5 authors * 4 book per author)
  + With all above pre-processing steps


## NLP Models and Techniques
