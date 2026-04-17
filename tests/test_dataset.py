import pytest
import yaml
import torch
from src.dataset import (generate_adult_dataset, build_vocab_maps, tokenize, NextWordDataset)
from torch.utils.data import random_split

@pytest.fixture
def config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_vocab_size(config):
    _, _, all_words = generate_adult_dataset(config['data']['seed'])
    assert len(all_words) == 10000

def test_sentence_length_distribution(config):
    short_corpus, long_corpus, _ = generate_adult_dataset(config['data']['seed'])
    for s in short_corpus[:100]:
        assert len(s) >= 5
        assert len(s) <= 9
    for s in long_corpus[:100]:
        assert len(s) >= 12
        assert len(s) <= 22

def test_train_test_split_ratio(config):
    total = 100
    train_size = int(config['data']['train_ratio'] * total)
    val_size = int(config['data']['val_ratio'] * total)
    test_size = total - train_size - val_size
    
    assert train_size == 80
    assert val_size == 10
    assert test_size == 10

def test_dataset_samples():
    indexed_sentences = [[2, 3, 4, 5]] # 4 words
    pad_idx = 0
    window_size = 2
    ds = NextWordDataset(indexed_sentences, pad_idx, window_size)
    assert len(ds) == 3
    inp, tgt, s_len = ds[0]
    assert s_len == 4
    assert inp.tolist() == [0, 2]
    assert tgt.item() == 3
