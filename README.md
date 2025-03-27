# TGNEncoding

## File structure

- `Graphs/`: Contains the code for creating the graphs
- `Embedding/`: Contains the code for encoding the graphs
- `Test/`: Contains the code for testing the encoding

### Graphs

- `ba_graphs.py`: Contains the code for creating Barabasi-Albert graphs
- `create_tgns.py`: Contains the code for creating temporal graph networks from initial graphs

### Embedding

- `encoding1.py`: Contains the code for encoding the graphs using a simple encoding

### Test

- `dataset.py`: Contains the code for creating a dataset of the given graph, with the given tests

### Pipeline

- `run.py`: Contains the code for running the tests from the terminal

### Data

- `graphs/`: Contains the graphs used in the tests
- `embeddings/`: Contains the embeddings of the graphs
- `results/`: Contains the results of the tests

## Test configuration
Tests are defined in the following json format:
```json
{
    "questions": list[str],
    "true_answers": list[str]
}
```

- `type`: The type of test to run
- `config`: The configuration for the test
- `questions`: The questions to ask the model


## Pipeline
Usage:
```bash
python run.py --graph_type <graph_type> --graph_config <parameters> --samples <number_of_samples> --tests <test_bitstring>
```

- `graph_type`: The type of graph to create
- `graph_config`: The configuration for the graph
- `samples`: The number of samples to create
- `tests`: The tests to run

## Graphs



