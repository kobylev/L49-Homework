import torch
import torch.nn.functional as F
import yaml
import argparse
import os
from src.model import NextWordModel
from src.dataset import generate_adult_dataset, build_vocab_maps, UNK_TOKEN, PAD_TOKEN

def load_demo_resources(model_path, config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate vocab (deterministic based on seed)
    _, _, all_words = generate_adult_dataset(config['data']['seed'])
    word2idx, idx2word = build_vocab_maps(all_words)
    
    # Infer model parameters from filename if possible
    rnn_type = "LSTM" if "LSTM" in model_path else "RNN"
    
    model = NextWordModel(
        vocab_size=len(word2idx),
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        pad_idx=word2idx[PAD_TOKEN],
        rnn_type=rnn_type
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, word2idx, idx2word, device, config

def predict_next_words(text, model, word2idx, idx2word, device, top_k=5):
    words = text.lower().split()
    indexed = [word2idx.get(w, word2idx[UNK_TOKEN]) for w in words]
    input_tensor = torch.tensor([indexed], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits, _ = model(input_tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        
    top_probs, top_indices = torch.topk(probs, top_k)
    
    results = []
    for i in range(top_k):
        results.append({
            "word": idx2word[top_indices[i].item()],
            "probability": top_probs[i].item()
        })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Next-Word Prediction Demo")
    parser.add_argument("--model", type=str, default="output/models/best_LSTM_2_short.pt", help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default="the researcher discovered that", help="Input sentence fragment")
    parser.add_argument("--top_k", type=int, default=5, help="Number of predictions to show")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        # Try to find any model in output/models
        models = [f for f in os.listdir("output/models") if f.endswith(".pt")]
        if models:
            args.model = os.path.join("output/models", models[0])
            print(f"Requested model not found. Using available model: {args.model}")
        else:
            print("Error: No trained models found in output/models/. Please run training first.")
            exit(1)
            
    print(f"Loading model from {args.model}...")
    model, word2idx, idx2word, device, config = load_demo_resources(args.model)
    
    print(f"\nInput Fragment: \"{args.input}\"")
    predictions = predict_next_words(args.input, model, word2idx, idx2word, device, top_k=args.top_k)
    
    print("\nTop Predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. {pred['word']:<15} ({pred['probability']*100:>5.1f}%)")
