import pytest
from src.preprocessing import (generate_structured_vocabulary, build_vocab_maps, 
                               tokenize, PAD_TOKEN, UNK_TOKEN)

def test_tokenizer_consistency():
    vocab_pools = generate_structured_vocabulary(100, 42)
    word2idx, idx2word = build_vocab_maps(vocab_pools)
    sentence = [vocab_pools["subjects"][0], vocab_pools["verbs"][0]]
    indexed = tokenize([sentence], word2idx)[0]
    decoded = [idx2word[i] for i in indexed]
    assert decoded == sentence

def test_unk_token_handling():
    vocab_pools = generate_structured_vocabulary(100, 42)
    word2idx, _ = build_vocab_maps(vocab_pools)
    sentence = ["nonexistentword"]
    indexed = tokenize([sentence], word2idx)[0]
    assert indexed[0] == word2idx[UNK_TOKEN]

def test_padding_token():
    vocab_pools = generate_structured_vocabulary(100, 42)
    word2idx, _ = build_vocab_maps(vocab_pools)
    assert word2idx[PAD_TOKEN] == 0
