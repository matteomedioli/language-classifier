import pytest
import torch
import pandas as pd
import numpy as np
from data import preprocessing, CLS, SEP, PAD

@pytest.mark.preprocessing
@pytest.mark.parametrize('file_name', ["Language_Detection.csv"])
@pytest.mark.parametrize('max_len', [3, 16, 100])
def test_post_preprocessing_sentence_lenght(file_name, max_len):
    # Arrange
    source = pd.read_csv(file_name)
    # Act
    target, vocab = preprocessing(source, max_len)
    source.drop(source[source.duplicated()].index, axis=0, inplace=True)
    first_words = target["Text"].apply(lambda x: x[0])
    filter_pad = target["Text"].apply(lambda x: [word for word in x if word != vocab[PAD]])
    # Assert
    # All sentences pad to max_len
    assert (target['Text'].str.len() == max_len).all()
    # All sentences start with CLS
    assert (first_words==vocab[CLS]).all()
    # All sentences end with SEP (after PAD filter)
    assert (filter_pad.apply(lambda x: x[-1])==vocab[SEP]).all()
