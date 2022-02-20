# Prepositions Prediction - "The Hound of the Baskervilles"

The objective of this Notebook is to predict the original preposition for each instance of the masked token found in the full text of "The Hound of the Baskervilles" as best as possible via the use of a fine-tuned RoBERTa model.

However, this model will not access any part of the original text of "The Hound of the Baskervilles" for its training/fine-tuning process; instead, I will use another work of Sir Arthur Conan Doyle, "The Adventures of Sherlock Holmes", to fine tune the language model.

The full text of "The Hound of the Baskervilles", by Sir Arthur Conan Doyle, is available for download at https://www.gutenberg.org/ebooks/2852.txt.utf-8 .

The full text of "The Adventures of Sherlock Holmes", by Sir Arthur Conan Doyle, can be found in my github repo at https://raw.githubusercontent.com/ckenlam/Language-Model/main/hound-train.txt .

This fine-tuned model has an accuracy of 64%; its perplexity also slightly improved from 4.98 to 4.95.

# High-Level Methodology 
1. Load a pre-trained RoBERTa model with Huggingface for masked language modeling.
2. Use "The Adventures of Sherlock Holmes" full text as training data to fine-tune the RoBERTa model.
3. Mask all the prepositions in "The Adventures of Sherlock Holmes" and "The Hound of the Baskervilles".
4. Use the data collator DataCollatorForLanguageModeling to randomly mask 15% of the tokens in each batch of "The Adventures of Sherlock Holmes" texts during the fine-tuning process. The goal is to provide sufficient domain adaptation when running the model on "The Hound of the Baskervilles" after fine-tuning.
5. After fine-tuning, run each line of "The Hound of the Baskervilles" through the model and generate a preposition prediction for each masked token.
6. Count the number of correct predictions for each line of "The Hound of the Baskervilles"
7. Save the results as [nlu_challenge_results.csv](https://github.com/ckenlam/Challenge-Nuance-NLU-Prepositions/blob/main/nlu_challenge_results.csv).

# How to load the model
The model can be loaded through Huggingface:
```python
from transformers import TFAutoModelForMaskedLM

#load the model
model = TFAutoModelForMaskedLM.from_pretrained("ckenlam/nlu_sherlock_model_20220220")
mask_filler = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=1)

#test the model
test_phrase = 'the hound <mask> the baskervilles'
pred = mask_filler(test_phrase)
print(f"Input sentence: {test_phrase}")
print(f"Predicted sentence: {pred[0]['sequence']}")
```
> 'Input sentence: the hound <mask> the baskervilles'
  
> 'Predicted sentence: the hound of the baskervilles'
