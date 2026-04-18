import yaml
from src.dataset import generate_adult_dataset

def main():
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    print("Generating dataset (100,000 sentences, 10,000 vocab)...")
    short, long, vocab = generate_adult_dataset(config['data']['seed'])
    print(f"Success! Generated {len(short)} short sentences and {len(long)} long sentences.")
    print(f"Total vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    main()
