import requests
import json
from collections import deque
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
session = requests.Session()
redirects_cache = {}

def get_final_url(name: str) -> str:
    """
    Retrieve the final redirected title of a Wikipedia page.

    Parameters:
    - name (str): The initial title of the Wikipedia page to check for redirects.

    Returns:
    - str: The final title of the Wikipedia page after following redirects.
    """
    if name in redirects_cache:
        return redirects_cache[name]

    url = f'https://en.wikipedia.org/w/api.php?action=query&titles={name}&redirects&format=json'
    response = session.get(url)
    data = json.loads(response.text)
    
    final_title = data["query"]["redirects"][0]["to"] if "redirects" in data["query"] else name
    redirects_cache[name] = final_title
    return final_title



def heuristic_link_sort(links: list, goal_keywords: list, current_depth: int, max_depth: int) -> list:
    """
    Sorts a list of Wikipedia page titles based on their relevance to the goal keywords.

    Parameters:
    - links (list): A list of Wikipedia page titles.
    - goal_keywords (list): Keywords that represent the goal topic.
    - current_depth (int): The current depth in the search tree.
    - max_depth (int): The maximum allowed depth in the search tree.

    Returns:
    - list: A sorted list of Wikipedia page titles based on their relevance.
    """
    depth_factor = (max_depth - current_depth) / max_depth
    top_n = int(len(links) * depth_factor)

    goal_embedding = model.encode([' '.join(goal_keywords)], convert_to_tensor=True)
    link_embeddings = model.encode(links, convert_to_tensor=True)

    cosine_scores = cosine_similarity(goal_embedding, link_embeddings)[0]
    sorted_links_scores = sorted(zip(links, cosine_scores), key=lambda x: x[1], reverse=True)

    top_n = max(top_n, 5)
    return [link for link, score in sorted_links_scores[:top_n]]



def get_wiki_links(title: str) -> list:
    """
    Retrieves all Wikipedia page titles linked from the given page.

    Parameters:
    - title (str): The title of the Wikipedia page to retrieve links from.

    Returns:
    - list: A list of titles of Wikipedia pages linked from the given page.
    """
    title = title.replace(' ', '_')
    url = f'https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=links&plnamespace=0&pllimit=max&format=json'
    response = session.get(url)
    page = json.loads(response.text)['query']['pages']
    pageid = next(iter(page))
    links = page[pageid].get('links', [])
    return [link['title'] for link in links]



def find_path(start_title: str, goal_title: str, max_depth: int = 3) -> tuple:
    """
    Finds a path from the start Wikipedia page title to the goal Wikipedia page title.

    Parameters:
    - start_title (str): The title of the starting Wikipedia page.
    - goal_title (str): The title of the goal Wikipedia page.
    - max_depth (int, optional): The maximum depth to search. Default is 3.

    Returns:
    - tuple: A tuple containing the path as a list and the size of the path as an integer.
    """
    normalized_goal_title = get_final_url(goal_title)
    start_title_links = get_wiki_links(start_title)
    sorted_start_links = heuristic_link_sort(start_title_links, normalized_goal_title.split(), 0, max_depth)
    
    for first_link_title in sorted_start_links:
        queue = deque([(first_link_title, [start_title, first_link_title], 1)])  
        visited = set([start_title])  
        
        while queue:
            current_title, path, current_depth = queue.popleft()
            if current_depth < max_depth:
                links = get_wiki_links(current_title)
                if links:  
                    sorted_links = heuristic_link_sort(links, normalized_goal_title.split(), current_depth, max_depth)
                    for link in sorted_links:
                        normalized_link_title = get_final_url(link)
                        if normalized_link_title == normalized_goal_title:
                            return path + [normalized_link_title], len(path) + 1
                        if normalized_link_title not in visited:
                            visited.add(normalized_link_title)
                            queue.append((normalized_link_title, path + [normalized_link_title], current_depth + 1))
                else:
                    print(f"No further links found from {current_title}.")
            else:
                break  
    
    return None, 1

if __name__ == "__main__":
    start_title = 'Nobel Prize'
    goal_title = 'Array (data structure)'
    path, size = find_path(start_title, goal_title)
    print(f"Path: {path}")
    print(f"Number of steps: {size - 1}")
