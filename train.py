import argparse
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk import word_tokenize as eng_tokenize
from langdetect import detect
from pythainlp.tokenize import word_tokenize as thai_tokenize
from nltk.stem.porter import PorterStemmer
from model import NeuralNet  # Ensure this is your model class
from nltk_utils import bag_of_words, stem  # Adjust if necessary

# Initialize stemmer
stemmer = PorterStemmer()

# Function to tokenize sentences based on detected language
def tokenize(sentence):
    lang = detect(sentence)
    if lang == 'th':  # Thai
        return thai_tokenize(sentence, engine='newmm')
    else:  # Default to English
        return eng_tokenize(sentence)
    

def main_train(args):
    jsfilename = args.jsfilename
    jsformat = "pretrained/" + jsfilename + ".json"

    # Load JSON data
    with open(jsformat, 'r', encoding='utf-8') as file:
        intents = json.load(file)

    # Data preparation
    all_words = []
    tags = []
    xy = []
    responses_map = {}

    # Process intents
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        responses_map[tag] = {}  # Store responses per language

        for lang, data in intent['language'].items():
            # Process patterns
            for pattern in data['patterns']:
                tokens = tokenize(pattern)
                tokens = [stem(word) for word in tokens]
                all_words.extend(tokens)
                xy.append((tokens, tag))

            # Map responses
            responses_map[tag][lang] = data['responses']

    # Stemming, removing duplicates, and sorting vocabulary
    ignore_words = ['?', '.', '!', ',', "'"]
    all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
    tags = sorted(set(tags))

    print(len(xy), "patterns")
    print(len(tags), "tags:", tags)
    print(len(all_words), "unique stemmed words:", all_words)

    # Prepare training data
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)


    # Model hyperparameters from argparse
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    hidden_size = args.hidden_size

    input_size = len(X_train[0])
    output_size = len(tags)

    # Dataset class
    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    # Data loader
    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=0)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(device, dtype=torch.long)

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final loss: {loss.item():.4f}')

    # Save model and metadata
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
        "responses_map": responses_map
    }
    FILE = "pretrained/model/" + jsfilename + '.pth'
    torch.save(data, FILE)
    print(f"Training complete. Model saved to {FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a chatbot model using intents JSON file")
    parser.add_argument('--jsfilename', type=str, default="",
                        help="The base filename (without extension) of the intents JSON file")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument('--hidden_size', type=int, default=16, help="Number of hidden units in the model")

    args = parser.parse_args()
    main_train(args)


## python train.py --jsfilename pretrained/intent_20241128.json --epochs 1000 --batch_size 8 --learning_rate 0.01 --hidden_size 16
## python train.py --jsfilename intent_20241128 --epochs 1000 --batch_size 8 --learning_rate 0.01 --hidden_size 16