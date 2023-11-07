# Wikipedia Path Finder

This Python script uses the Wikipedia API and machine learning to find semantically connected paths between Wikipedia articles. It's an effective demonstration of optimizing API calls, potentially useful for machine learning training datasets requiring precise data relationships. While initially designed for exploration and demonstration, with the right modifications, this script can become useful in constructing robust datasets and supporting various machine learning training that require understanding and retrieval of information.

## How It Works

The script operates through several key functions:

- **API Communication:** Utilizes `requests` to interact with the Wikipedia API.
- **NLP Model:** Employs the `SentenceTransformer` model for advanced natural language processing.
- **Caching Mechanism:** Implements caching to efficiently store and retrieve processed data, minimizing API calls.
- **GPU Acceleration:** Optionally uses GPU resources for faster processing of machine learning tasks when available.

## Detailed Functionality

- **Model Loading:** A pre-trained NLP model from Sentence Transformers is loaded (in this case BAAI/bge-base-en-v1.5), which might take some time during the initial run but is cached for subsequent executions.

- **heuristic_link_sort Function:** Includes an optional GPU parameter to enable hardware acceleration. It sorts links from a Wikipedia page by relevance using NLP and FAISS, adjusted by the current depth within the search tree to refine link prioritization.

- **find_path Function:** The core algorithm, implementing a breadth-first search to find a path between two Wikipedia articles. It uses `heuristic_link_sort` to decide which links to explore further, with optional GPU acceleration to speed up the process.

## Installation

To run this script, you will need Python and the following packages:

```bash
pip install requests
pip install sentence-transformers
pip install faiss-cpu # or faiss-gpu for GPU support
```

## Usage

To use the script, run the find_path function with the start and goal Wikipedia titles. For example:

python

```bash
start_title = 'Nobel Prize' 
goal_title = 'Array (data structure)'
# Optionally specify a GPU device ID for acceleration
path, size = find_path(start_title, goal_title, GPU=<GPU_DEVICE_ID>)
print(f"Path: {path}")
print(f"Number of steps: {size}")
```

If the script does not find a path, increase the `max_depth` as 2 is the default. Specify the GPU parameter if you want to use GPU acceleration.

## Model Caching

On its first run, the script downloads the necessary language model for NLP. This process may take some time but is only required once; the model will be cached for future use.


## License

This project is licensed under the MIT License. For more details, see the LICENSE file in the repository.
