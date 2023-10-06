import spacy
from utils.data_processing import (classify_author, clean_lexile, encode_author_type,
                                   clean_grade, group_and_encode_genre)
import pandas as pd


def data_preparation() -> None:
    spacy_ner_model = spacy.load('en_core_web_sm')
    prompt_grade = pd.read_csv(r'../../data/commonlit_texts.csv')
    prompt_grade['author_type'] = prompt_grade['author'].apply(lambda x: classify_author(spacy_ner_model, x))
    prompt_grade['lexile_md'] = prompt_grade['lexile'].apply(clean_lexile)
    prompt_grade = encode_author_type(prompt_grade)
    prompt_grade = clean_grade(prompt_grade)
    prompt_grade = group_and_encode_genre(prompt_grade)
    prompt_grade.to_csv("../../data/prompt_grade.csv", index=False)


if __name__ == "__main__":
    data_preparation()

