import yaml
from src.dataset import generate_adult_dataset, build_vocab_maps, tokenize

def main():
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    print("Preprocessing data (Tokenization + Vocabulary indexing)...")
    short, _, vocab = generate_adult_dataset(config['data']['seed'])
    word2idx, idx2word = build_vocab_maps(vocab)
    
    # Tokenize a small sample to verify
    sample = short[:1]
    tokenized = tokenize(sample, word2idx)
    
    print(f"Vocabulary built: {len(word2idx)} tokens.")
    print(f"Sample tokenization: {sample[0]} -> {tokenized[0]}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
