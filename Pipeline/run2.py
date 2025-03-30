# to run encoding2 on ba1.pkl
# write this in terminal: 
# python Pipeline/run2.py --encode --input ba1.pkl


# to run encoding2 on complete1.pkl
# write this in terminal: 
# python Pipeline/run2.py --encode --input complete1.pkl

# to run encoding2 on er1.pkl
# write this in terminal: 
# python Pipeline/run2.py --encode --input er1.pkl

#!/usr/bin/env python3
import argparse
import sys
import os
import pickle
from pathlib import Path

# Add parent directory to path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from other directories
from Encoding.encoding2 import encoding2


def main():
    parser = argparse.ArgumentParser(description='Run TGN encoding2 on specified graph files')

    # Encoding options
    parser.add_argument('--encode', action='store_true', help='Encode a TGN using encoding2')
    parser.add_argument('--input', type=str, default='ba1.pkl', help='Input graph file')
    parser.add_argument('--output', type=str, help='Output file path')

    args = parser.parse_args()

    # Ensure folders exist
    os.makedirs('Data/graphs', exist_ok=True)
    os.makedirs('Data/embeddings', exist_ok=True)

    # Encode the graph
    if args.encode:
        print(f"Loading TGN from {args.input}")
        with open(args.input, 'rb') as f:
            tgn = pickle.load(f)

        embedding = encoding2(tgn)

        output_file = args.output if args.output else f"Data/embeddings/{Path(args.input).stem}_encoding2.txt"
        with open(output_file, 'w') as f:
            f.write(embedding)

        print(f"Encoding2 saved to {output_file}")
        print("\nEncoding output:")
        print(embedding)


if __name__ == "__main__":
    main()

