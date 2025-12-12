"""
Distributed Server with Raft Consensus
Integrates HNSW vector database with Raft for distributed consensus.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import time
from contextlib import asynccontextmanager
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.hnsw.graph import HNSWGraph
from core.storage.engine import StorageEngine
from distributed.raft.node import RaftNode, RequestVoteMessage, AppendEntriesMessage, LogEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class InsertRequest(BaseModel):
    vector: List[float] = Field(..., description="Vector to insert")
    id: Optional[int] = Field(None, description="Optional external ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InsertResponse(BaseModel):
    id: int
    success: bool
    message: str


class SearchRequest(BaseModel):
    query: List[float] = Field(..., description="Query vector")
    k: int = Field(10, description="Number of neighbors", ge=1, le=1000)
    ef: Optional[int] = Field(None, description="Search quality parameter")


class SearchResult(BaseModel):
    id: int
    distance: float
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    results: List[SearchResult]
    latency_ms: float
    total_searched: int


class DeleteRequest(BaseModel):
    id: int


class DeleteResponse(BaseModel):
    success: bool
    message: str


class StatsResponse(BaseModel):
    node_id: str
    is_leader: bool
    current_leader: Optional[str]
    total_vectors: int
    total_searches: int
    total_insertions: int
    raft_state: Dict[str, Any]
    index_stats: Dict[str, Any]
    uptime_seconds: float


class AppState:
    def __init__(self):
        self.graph: Optional[HNSWGraph] = None
        self.storage: Optional[StorageEngine] = None
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.raft: Optional[RaftNode] = None
        self.start_time: float = time.time()
        self.node_id: str = ""


state = AppState()


async def apply_command(command: Dict[str, Any]):
    try:
        cmd_type = command.get("type")

        if cmd_type == "insert":
            vector = np.array(command["vector"], dtype=np.float64)
            external_id = command.get("external_id")
            metadata = command.get("metadata", {})

            node_id = state.graph.insert(vector, external_id=external_id)

            if metadata:
                state.metadata_store[node_id] = metadata

            logger.debug(f"Applied insert command: id={node_id}")

        elif cmd_type == "delete":
            node_id = command["id"]
            success = state.graph.delete(node_id)

            if success and node_id in state.metadata_store:
                del state.metadata_store[node_id]

            logger.debug(f"Applied delete command: id={node_id}, success={success}")

    except Exception as e:
        logger.error(f"Error applying command: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Distributed VectorFlow server...")

    state.node_id = os.getenv("NODE_ID", "node1")
    node_port = int(os.getenv("NODE_PORT", "8001"))
    peers_env = os.getenv("PEERS", "")

    peer_addresses = {}
    if peers_env:
        for peer in peers_env.split(","):
            peer = peer.strip()
            if peer:
                peer_id, peer_port = peer.split(":")
                peer_addresses[peer_id] = f"http://localhost:{peer_port}"

    logger.info(f"Node ID: {state.node_id}, Port: {node_port}")
    logger.info(f"Peers: {peer_addresses}")

    state.storage = StorageEngine(f"./data/{state.node_id}")

    try:
        if state.storage.exists():
            logger.info("Loading existing graph from disk...")
            state.graph = await state.storage.load_graph(HNSWGraph)
            metadata = await state.storage.load_metadata()
            state.metadata_store = metadata.get("metadata_store", {})
            logger.info(f"Loaded graph with {len(state.graph.nodes)} vectors")
        else:
            logger.info("Initializing new graph...")
            state.graph = HNSWGraph(
                dim=128,
                M=16,
                ef_construction=200,
                ef_search=50,
                distance_metric="cosine"
            )
    except Exception as e:
        logger.warning(f"Could not load graph: {e}. Starting fresh.")
        state.graph = HNSWGraph(
            dim=128,
            M=16,
            ef_construction=200,
            ef_search=50,
            distance_metric="cosine"
        )

    state.raft = RaftNode(
        node_id=state.node_id,
        peer_addresses=peer_addresses,
        state_machine=apply_command
    )

    await state.raft.start()

    state.start_time = time.time()
    logger.info(f"Distributed VectorFlow server started on port {node_port}")

    yield

    logger.info("Shutting down Distributed VectorFlow server...")

    if state.raft:
        await state.raft.stop()

    if state.graph and state.storage:
        logger.info("Saving graph to disk...")
        await state.storage.save_graph(
            state.graph,
            metadata={"metadata_store": state.metadata_store}
        )

    logger.info("Shutdown complete")


app = FastAPI(
    title="Distributed VectorFlow API",
    description="High-performance distributed vector search with Raft consensus",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/raft/request_vote")
async def raft_request_vote(request: Request):
    data = await request.json()

    msg = RequestVoteMessage(
        term=data["term"],
        sender_id=data["sender_id"],
        candidate_id=data["candidate_id"],
        last_log_index=data["last_log_index"],
        last_log_term=data["last_log_term"]
    )

    response = state.raft.handle_request_vote(msg)

    return {
        "term": response.term,
        "vote_granted": response.vote_granted
    }


@app.post("/raft/append_entries")
async def raft_append_entries(request: Request):
    data = await request.json()

    msg = AppendEntriesMessage(
        term=data["term"],
        sender_id=data["sender_id"],
        leader_id=data["leader_id"],
        prev_log_index=data["prev_log_index"],
        prev_log_term=data["prev_log_term"],
        entries=[],
        leader_commit=data["leader_commit"]
    )

    msg.entries = [
        LogEntry(
            term=e["term"],
            index=e["index"],
            command=e["command"],
            timestamp=e.get("timestamp", time.time())
        )
        for e in data.get("entries", [])
    ]

    response = state.raft.handle_append_entries(msg)

    return {
        "term": response.term,
        "success": response.success,
        "match_index": response.match_index
    }


@app.post("/insert", response_model=InsertResponse)
async def insert_vector(request: InsertRequest):
    try:
        if not state.raft.state.value == "leader":
            if state.raft.current_leader:
                raise HTTPException(
                    status_code=307,
                    detail=f"Not leader. Current leader: {state.raft.current_leader}"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="No leader elected yet. Please retry."
                )

        vector = request.vector

        if len(state.graph.nodes) == 0:
            state.graph.dim = len(vector)

        if len(vector) != state.graph.dim:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension {len(vector)} does not match index dimension {state.graph.dim}"
            )
        command = {
            "type": "insert",
            "vector": vector,
            "external_id": request.id,
            "metadata": request.metadata
        }

        success = await state.raft.append_entry(command)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to replicate to majority"
            )

        node_id = request.id if request.id is not None else state.graph.current_id - 1

        return InsertResponse(
            id=node_id,
            success=True,
            message="Vector inserted and replicated successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Insert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    try:
        start_time = time.time()
        query = np.array(request.query, dtype=np.float64)

        if len(query) != state.graph.dim:
            raise HTTPException(
                status_code=400,
                detail=f"Query dimension {len(query)} does not match index dimension {state.graph.dim}"
            )

        results = state.graph.search(query, k=request.k, ef=request.ef)

        search_results = [
            SearchResult(
                id=node_id,
                distance=float(distance),
                metadata=state.metadata_store.get(node_id, {})
            )
            for node_id, distance in results
        ]

        latency = (time.time() - start_time) * 1000

        return SearchResponse(
            results=search_results,
            latency_ms=latency,
            total_searched=len(state.graph.nodes)
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete", response_model=DeleteResponse)
async def delete_vector(request: DeleteRequest):
    try:
        if not state.raft.state.value == "leader":
            if state.raft.current_leader:
                raise HTTPException(
                    status_code=307,
                    detail=f"Not leader. Current leader: {state.raft.current_leader}"
                )
            else:
                raise HTTPException(
                    status_code=503,
                    detail="No leader elected yet. Please retry."
                )

        command = {
            "type": "delete",
            "id": request.id
        }

        success = await state.raft.append_entry(command)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to replicate to majority"
            )

        return DeleteResponse(
            success=True,
            message="Vector deleted and replicated successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    try:
        index_stats = state.graph.get_stats()
        raft_state = state.raft.get_state()
        uptime = time.time() - state.start_time

        return StatsResponse(
            node_id=state.node_id,
            is_leader=raft_state["is_leader"],
            current_leader=state.raft.current_leader,
            total_vectors=len(state.graph.nodes),
            total_searches=state.graph.total_searches,
            total_insertions=state.graph.total_insertions,
            raft_state=raft_state,
            index_stats=index_stats,
            uptime_seconds=uptime
        )

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "node_id": state.node_id,
        "is_leader": state.raft.state.value == "leader",
        "raft_state": state.raft.state.value,
        "total_vectors": len(state.graph.nodes) if state.graph else 0
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("NODE_PORT", "8001"))
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    uvicorn.run(app, host="0.0.0.0", port=port, log_config=log_config)
