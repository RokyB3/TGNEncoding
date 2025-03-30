#!/usr/bin/env python3
import argparse
import sys
import os
import pickle
from pathlib import Path

# Add parent directory to path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from other directories
from Graphs.ba_create_tgns import create_tgn
from Graphs.ba_graphs import create_BA_graph
from Graphs.ba_step import StepConfig
from Encoding.encoding1 import encoding1

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run TGN encoding and testing pipeline')
    
    # Add arguments
    parser.add_argument('--create-graph', action='store_true', 
                        help='Create a new Barabasi-Albert graph')
    parser.add_argument('--nodes', type=int, default=5,
                        help='Number of nodes for BA graph (default: 5)')
    parser.add_argument('--degree', type=int, default=2,
                        help='Average degree for BA graph (default: 2)')
    
    parser.add_argument('--create-tgn', action='store_true',
                        help='Create a temporal graph network from a base graph')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations for TGN (default: 3)')
    parser.add_argument('--n-add', type=int, default=2,
                        help='Number of nodes to add in each step (default: 2)')
    parser.add_argument('--p-add', type=float, default=0.5,
                        help='Probability of adding edges to new nodes (default: 0.5)')
    parser.add_argument('--p-remove', type=float, default=0.5,
                        help='Probability of removing edges (default: 0.5)')
    
    parser.add_argument('--encode', action='store_true',
                        help='Encode a TGN using the specified encoding method')
    parser.add_argument('--encoding-method', type=str, default='encoding1',
                        choices=['encoding1'], 
                        help='Encoding method to use (default: encoding1)')
    
    parser.add_argument('--input', type=str, default='ba1.pkl',
                        help='Input file path (default: ba1.pkl)')
    parser.add_argument('--output', type=str, 
                        help='Output file path (default: based on operation)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create data directories if they don't exist
    os.makedirs('Data/graphs', exist_ok=True)
    os.makedirs('Data/embeddings', exist_ok=True)
    os.makedirs('Data/results', exist_ok=True)
    
    # Process commands
    if args.create_graph:
        print(f"Creating BA graph with {args.nodes} nodes and degree {args.degree}")
        ba_graph = create_BA_graph(args.nodes, args.degree)
        
        # Determine output file
        output_file = args.output if args.output else f"Data/graphs/ba_n{args.nodes}_m{args.degree}.pkl"
        
        # Save the graph
        with open(output_file, 'wb') as f:
            pickle.dump(ba_graph, f)
        print(f"Graph saved to {output_file}")
        
    if args.create_tgn:
        # Load base graph if not creating one
        if not args.create_graph:
            print(f"Loading base graph from {args.input}")
            with open(args.input, 'rb') as f:
                ba_graph = pickle.load(f)
        
        # Configure step parameters
        step_config = StepConfig(n_add=args.n_add, p_add=args.p_add, p_remove=args.p_remove)
        
        print(f"Creating TGN with {args.iterations} iterations")
        tgn = create_tgn(ba_graph, step_config, args.iterations)
        
        # Determine output file
        output_file = args.output if args.output else f"Data/graphs/tgn_i{args.iterations}.pkl"
        
        # Save the TGN
        with open(output_file, 'wb') as f:
            pickle.dump(tgn, f)
        print(f"TGN saved to {output_file}")
    
    if args.encode:
        # Load TGN
        print(f"Loading TGN from {args.input}")
        with open(args.input, 'rb') as f:
            tgn = pickle.load(f)
        
        # Encode based on selected method
        if args.encoding_method == 'encoding1':
            embedding = encoding1(tgn)
            
        # Determine output file
        output_file = args.output if args.output else f"Data/embeddings/{Path(args.input).stem}_{args.encoding_method}.txt"
        
        # Save the embedding
        with open(output_file, 'w') as f:
            f.write(embedding)
        print(f"Encoding saved to {output_file}")
        
        # Also print the encoding
        print("\nEncoding output:")
        print(embedding)

if __name__ == "__main__":
    main()
