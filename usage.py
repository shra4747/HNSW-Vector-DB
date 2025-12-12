"""
Simple usage example
"""

import requests
import numpy as np
import random

NODES = ["http://localhost:8001", "http://localhost:8002", "http://localhost:8003"]

def find_leader():
    for node_url in NODES:
        try:
            response = requests.get(f"{node_url}/health", timeout=2)
            health = response.json()
            if health.get("is_leader"):
                return node_url
        except Exception as e:
            print(f"Could not reach {node_url}: {e}")
    return None

def find_random_follower(leader_url):
    followers = [url for url in NODES if url != leader_url]

    if not followers:
        return leader_url

    follower_url = random.choice(followers)
    return follower_url

leader_url = find_leader()
if not leader_url:
    exit(1)

# Insert a vector (must go to leader)
vector = np.random.randn(128).tolist()
try:
    response = requests.post(f"{leader_url}/insert", json={
        "vector": vector,
        "metadata": {"description": "example vector from usage.py"}
    }, timeout=10)
except Exception as e:
    print(f"Error during insert: {e}")
    exit(1)

follower_url = find_random_follower(leader_url)
query = np.random.randn(128).tolist()
try:
    response = requests.post(f"{follower_url}/search", json={
        "query": query,
        "k": 10
    }, timeout=5)

    print(response.json())
except Exception as e:
    print(f"❌ Error during search: {e}")
