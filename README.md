# HNSW Vector Database with Raft Consensus

A vector database built from scratch that combines the HNSW algorithm for fast approximate nearest neighbor search with Raft consensus for distributed fault tolerance.

## What it does

This database stores high-dimensional vectors and lets you search for similar ones quickly. It's distributed across multiple nodes using Raft consensus, so if one server goes down, the system keeps working.

**Example use case:** Store a multitude of feature films as 128-dimensional vectors and quickly find movies with similar characteristics by searching for the closest vectors in the database.

## Key Features

**HNSW Search**
- Sub-millisecond similarity search on large datasets
- Probabilistic multi-layer graph structure (like a skip linked list)
- Uses euclidean, cosine, manhattan, and dot product distances

**Raft Consensus**
- Distributed across 3+ nodes for fault tolerance
- Leader election with automatic failover
- Strong consistency - all nodes see the same data
- Tolerates (N-1)/2 failures (e.g., 2 failures in a 5-node cluster)

**REST API**
- FastAPI server with async support
- Insert, search, delete operations
- Real-time cluster statistics

## Quick Start

```bash
# Start a 3-node cluster
./start_cluster.sh --reset

# Or start 5 nodes
./start_cluster.sh 5 --reset
```
## How it works

### HNSW Algorithm
HNSW creates a hierarchical graph where each vector is a node. Instead of comparing your query to every vector (slow), it navigates through layers of the graph, only checking neighbors at each step. This gives O(log N) search time instead of O(N).

### Raft Consensus
When you insert a vector:
1. Request goes to the leader node
2. Leader adds it to its log and replicates to followers
3. Once a majority confirms (2/3 nodes, 3/5 nodes, etc.), it's committed
4. All nodes apply the insert to their local HNSW index

Searches can happen on any node since all nodes have the same data.

## Architecture

```
Client
  │
  ├─> Node 1 (Follower)  ─┐
  ├─> Node 2 (Leader)     ├─> Raft Consensus (HTTP)
  └─> Node 3 (Follower)  ─┘

Each node has:
- FastAPI server
- Raft consensus layer
- HNSW vector index
- Storage engine
```

## Tech Stack

- **Python** for everything
- **NumPy** for vector operations
- **FastAPI** for the REST API
- **httpx** for node-to-node communication
- **msgpack** for serialization