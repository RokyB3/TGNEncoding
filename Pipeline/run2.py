#!/usr/bin/env python3
import sys
import os
import pickle
from pathlib import Path

# Add parent directory to path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import encoding2
from Encoding.encoding2 import encoding2

def encode_and_save(file_path, encoding_func):
    print(f"Loading TGN from {file_path}")
    with open(file_path, 'rb') as f:
        tgn = pickle.load(f)

    embedding = encoding_func(tgn)

    output_path = f"Data/embeddings/{Path(file_path).stem}_encoding2.txt"
    with open(output_path, 'w') as f:
        f.write(embedding)

    print(f"Encoding2 saved to {output_path}")
    print("\nEncoding output:")
    print(embedding)
    print("\n" + "-" * 80 + "\n")

def main():
    os.makedirs('Data/embeddings', exist_ok=True)

    # All TGN graph files to encode
    graph_files = [
        "ba1.pkl", "ba2.pkl", "ba3.pkl", "ba4.pkl",
        "complete1.pkl", "complete2.pkl", "complete3.pkl", "complete4.pkl",
        "er1.pkl", "er2.pkl", "er3.pkl", "er4.pkl"
    ]

    for file in graph_files:
        if os.path.exists(file):
            encode_and_save(file, encoding2)
        else:
            print(f"⚠️ File {file} not found. Skipping.")

if __name__ == "__main__":
    main()


