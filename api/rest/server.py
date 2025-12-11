"""
FastAPI REST API Server
High-performance REST API for vector operations with async support.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import time
from contextlib import asynccontextmanager
import logging

from core.hnsw.graph import HNSWGraph
from core.storage.engine import StorageEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    total_vectors: int
    total_searches: int
    total_insertions: int
    index_stats: Dict[str, Any]
    uptime_seconds: float

class BatchInsertRequest(BaseModel):
    vectors: List[List[float]]
    ids: Optional[List[int]] = None
    metadata: List[Dict[str, Any]] = []

class BatchInsertResponse(BaseModel):
    ids: List[int]
    total_inserted: int
    failed: int
    latency_ms: float

class AppState:
    def __init__(self):
        self.graph: Optional[HNSWGraph] = None
        self.storage: Optional[StorageEngine] = None
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.start_time: float = time.time()

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting VectorFlow API server...")
    state.storage = StorageEngine("./data/vectorflow")
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
    state.start_time = time.time()
    logger.info("VectorFlow API server started successfully")
    yield
    logger.info("Shutting down VectorFlow API server...")
    if state.graph and state.storage:
        logger.info("Saving graph to disk...")
        await state.storage.save_graph(
            state.graph,
            metadata={"metadata_store": state.metadata_store}
        )
    logger.info("Shutdown complete")

app = FastAPI(
    title="VectorFlow API",
    description="High-performance distributed vector search engine with HNSW",
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

@app.post("/insert", response_model=InsertResponse)
async def insert_vector(request: InsertRequest):
    try:
        vector = np.array(request.vector, dtype=np.float64)
        if len(state.graph.nodes) == 0:
            state.graph.dim = len(vector)
        if len(vector) != state.graph.dim:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension {len(vector)} does not match index dimension {state.graph.dim}"
            )
        node_id = state.graph.insert(vector, external_id=request.id)
        if request.metadata:
            state.metadata_store[node_id] = request.metadata
        return InsertResponse(
            id=node_id,
            success=True,
            message="Vector inserted successfully"
        )
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
        success = state.graph.delete(request.id)
        if success and request.id in state.metadata_store:
            del state.metadata_store[request.id]
        return DeleteResponse(
            success=success,
            message="Vector deleted successfully" if success else "Vector not found"
        )
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_insert", response_model=BatchInsertResponse)
async def batch_insert_vectors(request: BatchInsertRequest):
    try:
        start_time = time.time()
        ids = []
        failed = 0
        for i, vector_list in enumerate(request.vectors):
            try:
                vector = np.array(vector_list, dtype=np.float64)
                if len(state.graph.nodes) == 0:
                    state.graph.dim = len(vector)
                external_id = request.ids[i] if request.ids and i < len(request.ids) else None
                node_id = state.graph.insert(vector, external_id=external_id)
                if request.metadata and i < len(request.metadata):
                    state.metadata_store[node_id] = request.metadata[i]
                ids.append(node_id)
            except Exception as e:
                logger.warning(f"Failed to insert vector {i}: {e}")
                failed += 1
        latency = (time.time() - start_time) * 1000
        return BatchInsertResponse(
            ids=ids,
            total_inserted=len(ids),
            failed=failed,
            latency_ms=latency
        )
    except Exception as e:
        logger.error(f"Batch insert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    try:
        index_stats = state.graph.get_stats()
        uptime = time.time() - state.start_time
        return StatsResponse(
            total_vectors=len(state.graph.nodes),
            total_searches=state.graph.total_searches,
            total_insertions=state.graph.total_insertions,
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
        "version": "0.1.0",
        "total_vectors": len(state.graph.nodes) if state.graph else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
