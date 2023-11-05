# Text Detoxification

Constantine Smirnov

B20-RO-01

k.smirnov@innopolis.university

## Description

The project explores the approaches to perform text detoxification.

The baseline model simply replaces or removes toxic words using a dictionary,
the main approach finetunes T5-small detoxificate text.

## Usage

### Dataset

Download the dataset by running `src/data/make_dataset.py` **from the root of the
project!**

Or download the [dataset](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip) manually, unzip it, and put it into `data/raw/`

### Dependencies

Install requirements.txt:

`pip install -r requirements.txt`

### Detoxification

To see the detoxified versions of inputs, run the following from the root of the project:

```
python src/models/predict_model.py src/models/input.txt
```

**Make sure you have installed all deps**

You can edit `input.txt` or use your own text

## Project Structure

### Reports

`reports/` folder contains two reports describing ideas and exploration process.

### Notebooks

`notebooks/` folder contains all notebooks, which include:

- Data exploration notebook
- Baseline solution notebook
- Solution using a pre-trained Bert
- Attempt at fine-tuning a t5 pre-trained model

### Src

`src/` folder contains `python` scripts that can be executed to conduct training and prediction.




## References

- [Александр Панченко — Monolingual and Cross-lingual Text Detoxification](https://www.youtube.com/watch?v=PEo3UJKwsN0&t=1219s&ab_channel=%D0%9C%D0%A2%D0%A1Digital)
- [Monolingual and Cross-lingual Text Detoxification [in Russian]](https://www.youtube.com/watch?v=1RsHbmzY2Mg&ab_channel=BayesGroup.ru)
- [Text Detoxification using Large Pre-trained Neural Models](https://arxiv.org/abs/2109.08914)
- [GitHub repository related to the paper right above](https://github.com/s-nlp/detox)
