"""
Storage Engine for Vector Persistence
Handles serialization, deserialization, and disk persistence of HNSW graphs.
"""

import numpy as np
import msgpack
import aiofiles
import os
from typing import Dict, Any, Optional
from pathlib import Path
import json
import struct

class StorageEngine:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.msgpack"
        self.metadata_file = self.storage_path / "metadata.json"
        self.vectors_file = self.storage_path / "vectors.bin"

    async def save_graph(self, graph: Any, metadata: Optional[Dict] = None):
        graph_data = {
            "dim": graph.dim,
            "M": graph.M,
            "ef_construction": graph.ef_construction,
            "ef_search": graph.ef_search,
            "ml": graph.ml,
            "max_elements": graph.max_elements,
            "entry_point": graph.entry_point,
            "current_id": graph.current_id,
            "nodes": {}
        }

        vectors_data = []
        for node_id, node in graph.nodes.items():
            graph_data["nodes"][str(node_id)] = {
                "id": node.id,
                "level": node.level,
                "connections": {
                    str(level): list(conns)
                    for level, conns in node.connections.items()
                },
                "vector_offset": len(vectors_data)
            }
            vectors_data.append(node.vector)

        async with aiofiles.open(self.index_file, 'wb') as f:
            await f.write(msgpack.packb(graph_data, use_bin_type=True))

        if vectors_data:
            vectors_array = np.vstack(vectors_data)
            async with aiofiles.open(self.vectors_file, 'wb') as f:
                await f.write(struct.pack('II', *vectors_array.shape))
                await f.write(vectors_array.tobytes())

        if metadata is None:
            metadata = {}
        metadata.update({
            "total_nodes": len(graph.nodes),
            "dimension": graph.dim,
            "total_searches": graph.total_searches,
            "total_insertions": graph.total_insertions
        })

        async with aiofiles.open(self.metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))

    async def load_graph(self, graph_class: Any, **graph_kwargs) -> Any:
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_file}")

        async with aiofiles.open(self.index_file, 'rb') as f:
            content = await f.read()
            graph_data = msgpack.unpackb(content, raw=False)

        async with aiofiles.open(self.vectors_file, 'rb') as f:
            shape_bytes = await f.read(8)
            rows, cols = struct.unpack('II', shape_bytes)
            data_bytes = await f.read()
            vectors_array = np.frombuffer(data_bytes, dtype=np.float64).reshape(rows, cols)

        graph_init_params = {
            "dim": graph_data["dim"],
            "M": graph_data["M"],
            "ef_construction": graph_data["ef_construction"],
            "ef_search": graph_data["ef_search"],
            "ml": graph_data["ml"],
            "max_elements": graph_data["max_elements"]
        }
        graph_init_params.update(graph_kwargs)

        graph = graph_class(**graph_init_params)
        graph.entry_point = graph_data["entry_point"]
        graph.current_id = graph_data["current_id"]

        from core.hnsw.graph import HNSWNode
        from collections import defaultdict

        for node_id_str, node_data in graph_data["nodes"].items():
            node_id = int(node_id_str)
            vector = vectors_array[node_data["vector_offset"]]
            connections = defaultdict(set)
            for level_str, conns in node_data["connections"].items():
                connections[int(level_str)] = set(conns)
            node = HNSWNode(
                id=node_id,
                vector=vector,
                level=node_data["level"],
                connections=dict(connections)
            )
            graph.nodes[node_id] = node

        return graph

    async def load_metadata(self) -> Dict[str, Any]:
        if not self.metadata_file.exists():
            return {}

        async with aiofiles.open(self.metadata_file, 'r') as f:
            content = await f.read()
            return json.loads(content)

    def exists(self) -> bool:
        return self.index_file.exists() and self.vectors_file.exists()
