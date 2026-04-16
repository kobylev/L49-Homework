import random
from typing import Dict, List, Tuple
from utils.config import PAD_TOKEN, UNK_TOKEN

def generate_vocabulary(vocab_size: int, seed: int) -> List[str]:
    """
    Build `vocab_size` unique pseudo-English words via combinatorial
    phoneme concatenation.  Sorted output guarantees reproducibility.
    """
    consonants = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m",
                  "n", "p", "r", "s", "t", "v", "w", "z",
                  "br", "cr", "dr", "fl", "gr", "pr", "st", "tr"]
    vowels     = ["a", "e", "i", "o", "u", "ae", "ou", "ai", "ea"]
    endings    = ["", "s", "ed", "ing", "er", "ly", "tion", "al", "ous", "ment"]

    vocab: set = set()
    rng = random.Random(seed)
    while len(vocab) < vocab_size:
        word = (rng.choice(consonants) + rng.choice(vowels)
                + rng.choice(consonants) + rng.choice(vowels)
                + rng.choice(endings))
        vocab.add(word)

    vocab_list = sorted(vocab)
    print(f"[Data]  Vocabulary: {len(vocab_list):,} unique words generated.")
    return vocab_list

def build_vocab_maps(vocab_words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Index 0 -> <PAD>  (embedding kept at zero, ignored by loss)
    Index 1 -> <UNK>
    Index 2+ -> real words
    """
    word2idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w in vocab_words:
        word2idx[w] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}
    print(f"[Data]  Vocab maps ready. Total tokens (incl. specials): {len(word2idx):,}")
    return word2idx, idx2word

def generate_sentences(vocab_words: List[str],
                        num_sentences: int,
                        min_len: int,
                        max_len: int,
                        seed: int) -> List[List[str]]:
    """Randomly assemble sentences from the vocabulary."""
    rng = random.Random(seed + 1)
    sentences = [rng.choices(vocab_words, k=rng.randint(min_len, max_len))
                 for _ in range(num_sentences)]
    avg = sum(len(s) for s in sentences) / len(sentences)
    print(f"[Data]  {num_sentences:,} sentences generated. Avg length: {avg:.2f} words.")
    return sentences

def tokenise(sentences: List[List[str]],
             word2idx: Dict[str, int]) -> List[List[int]]:
    """Word strings -> integer index sequences (unknown words -> UNK)."""
    unk = word2idx[UNK_TOKEN]
    return [[word2idx.get(w, unk) for w in s] for s in sentences]
