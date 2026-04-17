import pytest
import yaml
import torch
from src.preprocessing import generate_structured_vocabulary, generate_structured_sentences, build_vocab_maps, tokenize
from src.dataset import NextWordDataset
from torch.utils.data import random_split

@pytest.fixture
def config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_vocab_size(config):
    vocab_pools = generate_structured_vocabulary(100, config['data']['seed'])
    word2idx, _ = build_vocab_maps(vocab_pools)
    assert len(word2idx) >= 100

def test_sentence_length_distribution(config):
    vocab_pools = generate_structured_vocabulary(100, config['data']['seed'])
    sentences = generate_structured_sentences(vocab_pools, 100, config['data']['seed'])
    for s in sentences:
        assert len(s) >= 3
        assert len(s) <= 4

def test_train_test_split_ratio(config):
    total = 100
    train_size = int(config['data']['train_ratio'] * total)
    val_size = int(config['data']['val_ratio'] * total)
    test_size = total - train_size - val_size
    
    assert train_size == 80
    assert val_size == 10
    assert test_size == 10

def test_no_leakage_between_splits():
    indexed_sentences = [[i, i+1] for i in range(100)]
    total = len(indexed_sentences)
    train_size = 80
    val_size = 10
    test_size = 10
    
    train_indices, val_indices, test_indices = random_split(
        range(total), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

def test_dataset_samples():
    indexed_sentences = [[2, 3, 4, 5]] # 4 words
    pad_idx = 0
    window_size = 2
    ds = NextWordDataset(indexed_sentences, pad_idx, window_size)
    assert len(ds) == 3
    inp, tgt = ds[0]
    assert inp.tolist() == [0, 2]
    assert tgt.item() == 3
