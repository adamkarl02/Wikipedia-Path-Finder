# Wikipedia Path Finder

This Python script is designed to navigate Wikipedia articles. Leveraging both the Wikipedia API and machine learning models, the script intelligently finds a path from one specified article to another. 

## How It Works

The script operates through several key functions:

- **API Communication:** Utilizes `requests` to interact with the Wikipedia API.
- **NLP Model:** Employs the `SentenceTransformer` model for advanced natural language processing.
- **Cosine Similarity:** Applies `cosine_similarity` from `scikit-learn` to measure the relevance of links.
- **Caching Mechanism:** Implements caching to efficiently store and retrieve processed data, minimizing API calls.

## Detailed Functionality

- **Import Statements:** The code imports the necessary libraries for HTTP requests, JSON handling, queue management, sentence embedding, and cosine similarity computations.

- **Model Loading:** A pre-trained NLP model from Sentence Transformers is loaded (in this case BAAI/bge-base-en-v1.5), which might take some time during the initial run but is cached for subsequent executions.

- **heuristic_link_sort Function:** Sorts links from a Wikipedia page by relevance using NLP and cosine similarity, adjusted by the current depth within the search tree to refine link prioritization.

- **find_path Function:** The core algorithm, implementing a breadth-first search to find a path between two Wikipedia articles. It uses `heuristic_link_sort` to decide which links to explore further.

## Installation

To run this script, you will need Python and the following packages:

```bash
pip install requests
pip install sentence-transformers
pip install scikit-learn
```

## Usage

To use the script, run the find_path function with the start and goal Wikipedia titles. For example:

python

```bash
start_title = 'Nobel Prize' 
goal_title = 'Array (data structure)'
path, size = find_path(start_title, goal_title)
print(f"Path: {path}")
print(f"Number of steps: {size}")
```

If the script does not find a path, increase the `max_depth` as 3 is the default.

## Model Caching

On its first run, the script downloads the necessary language model for NLP. This process may take some time but is only required once; the model will be cached for future use.


## License

This project is licensed under the MIT License. For more details, see the LICENSE file in the repository.
