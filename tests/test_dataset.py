import pytest
import yaml
from src.preprocessing import generate_structured_vocabulary, generate_structured_sentences, build_vocab_maps, tokenize
from src.dataset import NextWordDataset

@pytest.fixture
def config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_vocab_size(config):
    vocab_pools = generate_structured_vocabulary(100, config['data']['seed'])
    word2idx, _ = build_vocab_maps(vocab_pools)
    assert len(word2idx) >= 100

def test_sentence_length_distribution(config):
    vocab_pools = generate_structured_vocabulary(config['data']['vocab_size'], config['data']['seed'])
    sentences = generate_structured_sentences(vocab_pools, 100, config['data']['seed'])
    # Our grammar ensures a minimum length (subject + verb + object = 3)
    # and a maximum (det + adj + sub + adv + verb + det + adj + obj + mod = 9)
    for s in sentences:
        assert len(s) >= 3
        assert len(s) <= 9

def test_dataset_samples():
    indexed_sentences = [[2, 3, 4, 5]] # 4 words
    pad_idx = 0
    window_size = 2
    ds = NextWordDataset(indexed_sentences, pad_idx, window_size)
    # Samples should be:
    # 1. [0, 2] -> 3
    # 2. [2, 3] -> 4
    # 3. [3, 4] -> 5
    assert len(ds) == 3
    inp, tgt = ds[0]
    assert inp.tolist() == [0, 2]
    assert tgt.item() == 3
