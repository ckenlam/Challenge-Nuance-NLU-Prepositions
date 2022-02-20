# NLU-Prepositions-Challenge

The objective of this Notebook is to predict the original preposition for each instance of the masked token found in the full text of "The Hound of the Baskervilles" as best as possible via the use of a fine-tuned RoBERTa model.

However, this model will not access any part of the original text of "The Hound of the Baskervilles" for its training/fine-tuning process; instead, I will use another work of Sir Arthur Conan Doyle, "The Adventures of Sherlock Holmes", to fine tune the language model.

The full text of "The Hound of the Baskervilles", by Sir Arthur Conan Doyle, is available for download at https://www.gutenberg.org/ebooks/2852.txt.utf-8 .

The full text of "The Adventures of Sherlock Holmes", by Sir Arthur Conan Doyle, can be found in my github repo at https://raw.githubusercontent.com/ckenlam/Language-Model/main/hound-train.txt .
