import random
from typing import List, Dict, Tuple

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def generate_structured_vocabulary(vocab_size: int, seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    
    consonants = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "z"]
    vowels = ["a", "e", "i", "o", "u"]
    
    def gen_word():
        return rng.choice(consonants) + rng.choice(vowels) + rng.choice(consonants) + rng.choice(vowels) + rng.choice(consonants) + rng.choice(vowels)

    pools_sizes = {
        "subjects": 500,
        "verbs": 200,
        "objects": 300,
        "modifiers": 200
    }
    
    vocab_pools = {}
    all_words = set()
    
    for pool_name, size in pools_sizes.items():
        pool = set()
        while len(pool) < size:
            w = gen_word()
            if w not in all_words:
                pool.add(w)
                all_words.add(w)
        vocab_pools[pool_name] = sorted(list(pool))
        
    total_requested = vocab_size - 2
    current_count = sum(pools_sizes.values())
    
    extra_words = set()
    while len(extra_words) + current_count < total_requested:
        w = gen_word()
        if w not in all_words and w not in extra_words:
            extra_words.add(w)
    
    vocab_pools["extra"] = sorted(list(extra_words))
    return vocab_pools

def generate_structured_sentences(vocab_pools: Dict[str, List[str]], num_sentences: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed + 1)
    sentences = []
    
    subjects = vocab_pools["subjects"]
    verbs = vocab_pools["verbs"]
    objects = vocab_pools["objects"]
    modifiers = vocab_pools["modifiers"]
    
    for i in range(num_sentences):
        sentence = []
        # Use index-based selection to create deterministic patterns
        # so the model can actually learn which verb follows which subject
        sub_idx = rng.randint(0, len(subjects) - 1)
        subject = subjects[sub_idx]
        
        # Verb depends on subject
        verb = verbs[sub_idx % len(verbs)]
        
        # Object depends on verb
        obj = objects[(sub_idx + 1) % len(objects)]
        
        sentence.append(subject)
        sentence.append(verb)
        sentence.append(obj)
        
        # Modifier depends on object
        if i % 2 == 0:
            mod = modifiers[(sub_idx + 2) % len(modifiers)]
            sentence.append(mod)
            
        sentences.append(sentence)
        
    return sentences

def build_vocab_maps(vocab_pools: Dict[str, List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for pool_name in ["subjects", "verbs", "objects", "modifiers", "extra"]:
        if pool_name in vocab_pools:
            for w in vocab_pools[pool_name]:
                if w not in word2idx:
                    word2idx[w] = len(word2idx)
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

def tokenize(sentences: List[List[str]], word2idx: Dict[str, int]) -> List[List[int]]:
    unk_idx = word2idx[UNK_TOKEN]
    return [[word2idx.get(w, unk_idx) for w in s] for s in sentences]
