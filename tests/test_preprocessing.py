import pytest
import yaml
from src.dataset import (generate_adult_dataset, build_vocab_maps, 
                         tokenize, PAD_TOKEN, UNK_TOKEN)
from src.model import NextWordModel

@pytest.fixture
def config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_tokenizer_consistency():
    _, _, all_words = generate_adult_dataset(42)
    word2idx, idx2word = build_vocab_maps(all_words)
    sentence = [all_words[10], all_words[20]]
    indexed = tokenize([sentence], word2idx)[0]
    decoded = [idx2word[i] for i in indexed]
    assert decoded == sentence

def test_unk_token_handling():
    _, _, all_words = generate_adult_dataset(42)
    word2idx, _ = build_vocab_maps(all_words)
    sentence = ["thisisdefinitelynotawordinthevocabulary123"]
    indexed = tokenize([sentence], word2idx)[0]
    assert indexed[0] == word2idx[UNK_TOKEN]

def test_embedding_shape(config):
    vocab_size = config['data'].get('vocab_size', 10000)
    model = NextWordModel(vocab_size, 16, 32, 1, 0.1, 0, "RNN")
    assert model.embedding.weight.shape == (vocab_size, 16)

def test_padding_token():
    _, _, all_words = generate_adult_dataset(42)
    word2idx, _ = build_vocab_maps(all_words)
    assert word2idx[PAD_TOKEN] == 0
