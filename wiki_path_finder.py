from heapq import nlargest
import requests
from collections import deque
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from functools import cache

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
session = requests.Session()

@cache
def get_wiki_data(name: str) -> dict:
    """
    Retrieve Wikipedia page data, including the final title after resolving redirects
    and a list of linked page titles from the given Wikipedia page name.

    Parameters:
    ----------
    name : str
        The Wikipedia page title to to retrieve data.

    Returns:
    -------
    dict
        A dictionary with two keys:
        'final_title': the final page title after following redirects.
        'links': a list of titles of linked Wikipedia pages.
    """
    name = name.replace(' ', '_')
    url = f'https://en.wikipedia.org/w/api.php?action=query&generator=links&titles={name}&redirects&gplnamespace=0&gpllimit=max&format=json'
    response = session.get(url)
    data = response.json()

    final_title = name
    if "redirects" in data["query"]:
        final_title = data["query"]["redirects"][-1]["to"]
    
    links_data = data['query'].get('pages', {})
    links_titles = [page['title'] for page_id, page in links_data.items() if 'title' in page]

    return {'final_title': final_title, 'links': links_titles}



def heuristic_link_sort(links: list, goal_keywords: list, current_depth: int, max_depth: int) -> list:
    """
    Sorts a list of Wikipedia page titles based on their relevance to the goal keywords,
    factoring in the current depth of the search to prioritize the most relevant links.

    Parameters:
    ----------
    links : list
        A list of Wikipedia page titles to be sorted.
    goal_keywords : list
        Keywords that represent the goal topic, used to determine relevance.
    current_depth : int
        The current depth in the search tree, used to adjust the number of links considered.
    max_depth : int
        The maximum allowed depth in the search tree, which influences the relevance sorting.

    Returns:
    -------
    list
        A sorted list of Wikipedia page titles based on their relevance to the goal keywords.
    """
    depth_factor = (max_depth - current_depth) / max_depth
    top_n = int(len(links) * depth_factor)
    top_n = max(top_n, 5)  

    goal_embedding = model.encode([' '.join(goal_keywords)], convert_to_tensor=True)
    link_embeddings = model.encode(links, convert_to_tensor=True)

    cosine_scores = cosine_similarity(goal_embedding, link_embeddings)[0]
    top_links_scores = nlargest(top_n, zip(links, cosine_scores), key=lambda x: x[1])
    return [link for link, score in top_links_scores]


def find_path(start_title: str, goal_title: str, max_depth: int = 3) -> tuple:
    """
    Attempts to find a path from a start Wikipedia page title to a goal Wikipedia page title
    by traversing linked pages using a breadth-first search approach, within a given depth.

    Parameters:
    ----------
    start_title : str
        The title of the starting Wikipedia page.
    goal_title : str
        The title of the goal Wikipedia page to find a path to.
    max_depth : int, optional
        The maximum depth to search in the link tree. Defaults to 3.

    Returns:
    -------
    tuple
        A tuple containing:
        - The path as a list of Wikipedia page titles from start to goal, if found.
        - The length of the path as an integer. If no path is found, returns None, 1.
    """
    normalized_goal_title = goal_title
    start_title_links = get_wiki_data(start_title)
    sorted_start_links = heuristic_link_sort(start_title_links['links'], normalized_goal_title.split(), 0, max_depth)
    
    for first_link_title in sorted_start_links:
        queue = deque([(first_link_title, [start_title, first_link_title], 1)])  
        visited = set([start_title])  
        
        while queue:
            current_title, path, current_depth = queue.popleft()
            print(path)
            if current_depth < max_depth:
                links = get_wiki_data(current_title)
                if links:  
                    sorted_links = heuristic_link_sort(links['links'], normalized_goal_title.split(), current_depth, max_depth)
                    for link in sorted_links:
                        if link == normalized_goal_title:
                            return path + [link], len(path) + 1
                        if link not in visited:
                            visited.add(link)
                            queue.append((link, path + [link], current_depth + 1))
                else:
                    print(f"No further links found from {current_title}.")
            else:
                break  
    
    return None, 1

if __name__ == "__main__":
    start_title = '2005 Azores subtropical storm'
    goal_title = 'Global Positioning System'
    path, size = find_path(start_title, goal_title, 2)
    print(f"Path: {path}")
    print(f"Number of steps: {size - 1}")
