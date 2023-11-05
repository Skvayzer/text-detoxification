
## Data exploration

### Dataset

### Issues with Translated Sentences

Contrary to expectations, the so-called "translated" sentences in the dataset do not consistently exhibit lower toxicity compared to the "reference" sentences. An investigation reveals that this assumption is not universally valid. Consequently, a reevaluation of toxicity for each entry is imperative, necessitating a dataset reconstruction.
Thats why I:
- Swaped reference/translation texts places if the toxicity score of the translation is higher than the score of the reference.
- I leave only data with high confidence of toxicity/non-toxicity (toxicity score > 0.99, non-toxicity score < 0.01>)

## Approaches

### Dictionary-based Toxic Word Removal/Replacement

This method, though straightforward to implement, might produce sentences with an artificial tone. Lacking machine learning components, it serves as a foundational solution.

Execution Steps:
1. Identify toxic words through dictionary lookup.
2. Eliminate or substitute these words with neutral alternatives

### Pre-trained T5
This method finetunes T5 on text-to-text pairs of toxic & non-toxic text.
