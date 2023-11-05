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


Project Structure

### Reports

`reports/` folder contains two reports describing ideas and exploration process.

### Notebooks

`notebooks/` folder contains all notebooks, which include:

- Data exploration notebook
- Baseline solution notebook
- Finetuning pretrained T5 model

### Src

`src/` folder contains `python` scripts that can be executed to conduct training and prediction.

## References

- [Text Detoxification using Large Pre-trained Neural Models](https://arxiv.org/abs/2109.08914)
- [Scoltech Paper Github Repository](https://github.com/s-nlp/detox)
