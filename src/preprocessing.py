import random
from typing import List, Dict, Tuple

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def generate_structured_vocabulary(vocab_size: int, seed: int) -> Dict[str, List[str]]:
    """
    Build structured vocabulary pools.
    """
    rng = random.Random(seed)
    
    # Define phonemes for generating pseudo-words
    consonants = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "z"]
    vowels = ["a", "e", "i", "o", "u"]
    
    def gen_word():
        return rng.choice(consonants) + rng.choice(vowels) + rng.choice(consonants) + rng.choice(vowels) + rng.choice(consonants) + rng.choice(vowels)

    # We need to fill vocab_size with structured words
    # Let's allocate proportions:
    # subjects: 20%, verbs: 15%, objects: 20%, adjectives: 15%, adverbs: 10%, determiners: 5%, modifiers: 15%
    
    pools = {
        "subjects": int(vocab_size * 0.20),
        "verbs": int(vocab_size * 0.15),
        "objects": int(vocab_size * 0.20),
        "adjectives": int(vocab_size * 0.15),
        "adverbs": int(vocab_size * 0.10),
        "determiners": int(vocab_size * 0.05),
        "modifiers": int(vocab_size * 0.15)
    }
    
    vocab_pools = {}
    all_words = set()
    
    for pool_name, size in pools.items():
        pool = set()
        while len(pool) < size:
            w = gen_word()
            if w not in all_words:
                pool.add(w)
                all_words.add(w)
        vocab_pools[pool_name] = list(pool)
        
    return vocab_pools

def generate_structured_sentences(vocab_pools: Dict[str, List[str]], num_sentences: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed + 1)
    sentences = []
    
    # Create "themes" or clusters to make it learnable
    num_clusters = 10
    clusters = []
    for _ in range(num_clusters):
        cluster = {
            "subjects": rng.sample(vocab_pools["subjects"], k=len(vocab_pools["subjects"]) // num_clusters),
            "verbs": rng.sample(vocab_pools["verbs"], k=len(vocab_pools["verbs"]) // num_clusters),
            "objects": rng.sample(vocab_pools["objects"], k=len(vocab_pools["objects"]) // num_clusters),
            "adjectives": rng.sample(vocab_pools["adjectives"], k=len(vocab_pools["adjectives"]) // num_clusters),
            "adverbs": rng.sample(vocab_pools["adverbs"], k=len(vocab_pools["adverbs"]) // num_clusters),
            "modifiers": rng.sample(vocab_pools["modifiers"], k=len(vocab_pools["modifiers"]) // num_clusters),
        }
        clusters.append(cluster)

    for _ in range(num_sentences):
        # Pick a cluster for this sentence
        c = rng.choice(clusters)
        sentence = []
        
        # Subject part
        if rng.random() > 0.3:
            sentence.append(rng.choice(vocab_pools["determiners"]))
        if rng.random() > 0.5:
            sentence.append(rng.choice(c["adjectives"]))
        sentence.append(rng.choice(c["subjects"]))
        
        # Verb part
        if rng.random() > 0.7:
            sentence.append(rng.choice(c["adverbs"]))
        sentence.append(rng.choice(c["verbs"]))
        
        # Object part
        if rng.random() > 0.4:
            sentence.append(rng.choice(vocab_pools["determiners"]))
        if rng.random() > 0.6:
            sentence.append(rng.choice(c["adjectives"]))
        sentence.append(rng.choice(c["objects"]))
        
        # Modifier
        if rng.random() > 0.7:
            sentence.append(rng.choice(c["modifiers"]))
            
        sentences.append(sentence)
        
    return sentences

def build_vocab_maps(vocab_pools: Dict[str, List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for pool in vocab_pools.values():
        for w in pool:
            if w not in word2idx:
                word2idx[w] = len(word2idx)
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

def tokenize(sentences: List[List[str]], word2idx: Dict[str, int]) -> List[List[int]]:
    unk_idx = word2idx[UNK_TOKEN]
    return [[word2idx.get(w, unk_idx) for w in s] for s in sentences]
