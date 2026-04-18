import torch
from torch.utils.data import Dataset
import random
from typing import List, Tuple, Dict, Set

# Constants
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"

class NextWordDataset(Dataset):
    """
    Converts sentences into multiple samples based on window size.
    """
    def __init__(self, indexed_sentences: List[List[int]], pad_idx: int, window_size: int):
        self.pad_idx = pad_idx
        self.window_size = window_size
        self.samples = []
        
        for sentence in indexed_sentences:
            s_len = len(sentence)
            if s_len < 2:
                continue
            for i in range(1, s_len):
                target = sentence[i]
                context = sentence[max(0, i - window_size):i]
                # Pad if context is shorter than window_size
                if len(context) < window_size:
                    padded_context = [pad_idx] * (window_size - len(context)) + context
                else:
                    padded_context = context
                self.samples.append((padded_context, target, s_len))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        inp, tgt, s_len = self.samples[idx]
        return (torch.tensor(inp, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long),
                s_len)

def generate_adult_dataset(seed: int = 42) -> Tuple[List[List[str]], List[List[str]], List[str]]:
    rng = random.Random(seed)
    
    # Syllable-based word generation
    cons = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "x", "z"]
    vows = ["a", "e", "i", "o", "u"]
    
    def gen_word():
        res = ""
        for _ in range(rng.randint(2, 4)):
            res += rng.choice(cons) + rng.choice(vows)
        return res

    all_words_set = set()
    pools_sizes = {
        "nouns": 2000,
        "verbs": 1500,
        "adverbs": 800,
        "adjectives": 800,
        "preps": 200,
        "academic": 240,
        "news": 240,
        "business": 240,
        "philosophy": 240,
        "everyday": 240,
        "function": 1500,
        "special": 200,
        "general": 10000 - (2000+1500+800+800+200+1200+1500+200) - 4 # = 1796
    }
    
    word_pools = {}
    for cat, size in pools_sizes.items():
        pool = []
        while len(pool) < size:
            w = gen_word()
            if w not in all_words_set:
                all_words_set.add(w)
                pool.append(w)
        word_pools[cat] = pool

    # Reserved tokens
    reserved = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
    for r in reserved:
        if r not in all_words_set:
            all_words_set.add(r)
    word_pools["special"] = word_pools["special"][:len(word_pools["special"])-len(reserved)] + reserved

    # Dependency generation
    def get_w(cat, idx, offset=0):
        p = word_pools.get(cat, word_pools.get("nouns"))
        return p[(idx + offset) % len(p)]

    def t1(): # Academic
        i = rng.randint(0, 10000)
        return ["the", get_w("nouns", i), get_w("verbs", i), get_w("adverbs", i+1), get_w("preps", i+2), "the", get_w("nouns", i+3)]
    def t1_b(): # Academic 2
        i = rng.randint(0, 10000)
        return ["recent", get_w("nouns", i), get_w("verbs", i), "that", get_w("nouns", i+1), get_w("verbs", i+2), get_w("adverbs", i+3)]
    def t2(): # News
        i = rng.randint(0, 10000)
        return ["the", get_w("news", i), get_w("news", i+1), get_w("verbs", i), get_w("news", i+2), get_w("preps", i), get_w("news", i+3)]
    def t2_b(): # News 2
        i = rng.randint(0, 10000)
        return ["analysts", get_w("verbs", i), "that", get_w("nouns", i), get_w("verbs", i+1), get_w("adverbs", i), get_w("preps", i+1), get_w("news", i)]
    def t3(): # Business
        i = rng.randint(0, 10000)
        return ["the", get_w("business", i), get_w("verbs", i), get_w("business", i+1), "by", get_w("business", i+2), get_w("preps", i), get_w("business", i+3)]
    def t3_b(): # Business 2
        i = rng.randint(0, 10000)
        return ["investors", get_w("verbs", i), get_w("business", i), get_w("adverbs", i), get_w("preps", i+1), "the", get_w("business", i+1)]
    def t4(): # Philosophy
        i = rng.randint(0, 10000)
        return ["the", get_w("philosophy", i), "of", get_w("nouns", i), get_w("verbs", i), get_w("adverbs", i), get_w("preps", i), get_w("philosophy", i+1)]
    def t4_b(): # Philosophy 2
        i = rng.randint(0, 10000)
        return ["one", get_w("verbs", i), "that", get_w("nouns", i), get_w("verbs", i+1), get_w("preps", i), get_w("philosophy", i)]
    def t5(): # Everyday
        i = rng.randint(0, 10000)
        return ["she", get_w("verbs", i), "the", get_w("nouns", i), get_w("adverbs", i), get_w("preps", i), get_w("everyday", i)]
    def t5_b(): # Everyday 2
        i = rng.randint(0, 10000)
        return ["they", get_w("verbs", i), "that", get_w("nouns", i), get_w("verbs", i+1), get_w("adverbs", i)]

    templates = [t1, t1_b, t2, t2_b, t3, t3_b, t4, t4_b, t5, t5_b]

    all_words_list = sorted(list(all_words_set))
    
    sentences_short = set()
    while len(sentences_short) < 80000:
        if rng.random() < 0.15:
            s = [rng.choice(all_words_list) for _ in range(rng.randint(5, 9))]
        else:
            s = rng.choice(templates)()
            if len(s) < 5: s += [rng.choice(word_pools["function"]) for _ in range(5 - len(s))]
            if len(s) > 9: s = s[:9]
        sentences_short.add(" ".join(s))

    sentences_long = set()
    while len(sentences_long) < 20000:
        if rng.random() < 0.15:
            s = [rng.choice(all_words_list) for _ in range(rng.randint(12, 22))]
        else:
            s1 = rng.choice(templates)()
            s2 = rng.choice(templates)()
            s = s1 + ["and"] + s2
            if len(s) < 12: s += [rng.choice(word_pools["function"]) for _ in range(12 - len(s))]
            if len(s) > 22: s = s[:22]
        sentences_long.add(" ".join(s))

    short_corpus = [s.split() for s in sentences_short]
    long_corpus = [s.split() for s in sentences_long]
    
    return short_corpus, long_corpus, all_words_list

def build_vocab_maps(all_words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    word2idx = {w: i for i, w in enumerate(all_words)}
    # Ensure PAD is 0 for embedding padding_idx
    if word2idx[PAD_TOKEN] != 0:
        old_zero_word = [w for w, i in word2idx.items() if i == 0][0]
        word2idx[old_zero_word] = word2idx[PAD_TOKEN]
        word2idx[PAD_TOKEN] = 0
    idx2word = {i: w for w, i in word2idx.items()}
    
    print(f"Vocabulary size confirmed: {len(word2idx)} tokens")
    assert len(word2idx) == 10000, f"Expected 10000 tokens, got {len(word2idx)}"
    
    return word2idx, idx2word

def tokenize(sentences: List[List[str]], word2idx: Dict[str, int]) -> List[List[int]]:
    unk_idx = word2idx.get(UNK_TOKEN, 1)
    return [[word2idx.get(w, unk_idx) for w in s] for s in sentences]
