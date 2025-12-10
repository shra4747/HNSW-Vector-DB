import numpy as np
import heapq
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import random
import math

@dataclass
class HNSWNode:
    id: int
    vector: np.ndarray
    level: int
    connections: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))

    def add_connection(self, level: int, neighbor_id: int):
        self.connections[level].add(neighbor_id)

    def get_connections(self, level: int) -> Set[int]:
        return self.connections.get(level, set())

class DistanceMetric:
    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot_product / (norm_a * norm_b))

    @staticmethod
    def manhattan(a: np.ndarray, b: np.ndarray) -> float:
        return np.sum(np.abs(a - b))

    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> float:
        return -np.dot(a, b)

class HNSWGraph:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        ml: float = 1.0 / math.log(2.0),
        distance_metric: str = "euclidean",
        max_elements: int = 1_000_000
    ):
        self.dim = dim
        self.M = M
        self.M_max = M
        self.M_max0 = M * 2
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml
        self.max_elements = max_elements

        self.distance_func = getattr(DistanceMetric, distance_metric)
        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point: Optional[int] = None
        self.current_id = 0

        self.lock = threading.RLock()

        self.total_searches = 0
        self.total_insertions = 0

    def _get_random_level(self) -> int:
        return int(-math.log(random.uniform(0, 1)) * self.ml)

    def _get_neighbors_heuristic(
        self,
        candidates: List[Tuple[float, int]],
        M: int,
        layer: int,
        extend_candidates: bool = True
    ) -> List[int]:
        candidates = sorted(candidates, key=lambda x: x[0])
        return [node_id for _, node_id in candidates[:M]]

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int
    ) -> List[Tuple[float, int]]:
        visited = set()
        candidates = []
        w = []

        for ep in entry_points:
            if ep not in self.nodes:
                continue
            dist = self.distance_func(query, self.nodes[ep].vector)
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(w, (-dist, ep))
            visited.add(ep)

        if not candidates:
            return []

        while candidates:
            current_dist, current = heapq.heappop(candidates)
            if current_dist > -w[0][0]:
                break
            node = self.nodes[current]
            for neighbor_id in node.get_connections(layer):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor = self.nodes[neighbor_id]
                    dist = self.distance_func(query, neighbor.vector)
                    if dist < -w[0][0] or len(w) < ef:
                        heapq.heappush(candidates, (dist, neighbor_id))
                        heapq.heappush(w, (-dist, neighbor_id))
                        if len(w) > ef:
                            heapq.heappop(w)
        return [(-dist, node_id) for dist, node_id in w]

    def insert(self, vector: np.ndarray, external_id: Optional[int] = None) -> int:
        with self.lock:
            if len(self.nodes) >= self.max_elements:
                raise ValueError(f"Maximum elements ({self.max_elements}) reached")
            if external_id is not None:
                node_id = external_id
                self.current_id = max(self.current_id, external_id + 1)
            else:
                node_id = self.current_id
                self.current_id += 1
            level = self._get_random_level()
            node = HNSWNode(id=node_id, vector=vector.copy(), level=level)
            self.nodes[node_id] = node
            if self.entry_point is None:
                self.entry_point = node_id
                self.total_insertions += 1
                return node_id
            entry_level = self.nodes[self.entry_point].level
            current_nearest = [self.entry_point]
            for lc in range(entry_level, level, -1):
                current_nearest = self._search_layer(vector, current_nearest, 1, lc)
                current_nearest = [node_id for _, node_id in current_nearest]
            for lc in range(level, -1, -1):
                candidates = self._search_layer(
                    vector, current_nearest, self.ef_construction, lc
                )
                M = self.M_max0 if lc == 0 else self.M_max
                neighbors = self._get_neighbors_heuristic(candidates, M, lc)
                for neighbor_id in neighbors:
                    node.add_connection(lc, neighbor_id)
                    self.nodes[neighbor_id].add_connection(lc, node_id)
                    neighbor_connections = self.nodes[neighbor_id].get_connections(lc)
                    if len(neighbor_connections) > M:
                        neighbor_candidates = [
                            (self.distance_func(self.nodes[neighbor_id].vector,
                                              self.nodes[c].vector), c)
                            for c in neighbor_connections
                        ]
                        new_connections = self._get_neighbors_heuristic(
                            neighbor_candidates, M, lc, extend_candidates=False
                        )
                        self.nodes[neighbor_id].connections[lc] = set(new_connections)
                current_nearest = neighbors
            if level > self.nodes[self.entry_point].level:
                self.entry_point = node_id
            self.total_insertions += 1
            return node_id

    def search(self, query: np.ndarray, k: int, ef: Optional[int] = None) -> List[Tuple[int, float]]:
        with self.lock:
            self.total_searches += 1
            if self.entry_point is None or len(self.nodes) == 0:
                return []
            if ef is None:
                ef = max(self.ef_search, k)
            current_nearest = [self.entry_point]
            entry_level = self.nodes[self.entry_point].level
            for lc in range(entry_level, 0, -1):
                current_nearest = self._search_layer(query, current_nearest, 1, lc)
                current_nearest = [node_id for _, node_id in current_nearest]
            current_nearest = self._search_layer(query, current_nearest, ef, 0)
            current_nearest = sorted(current_nearest, key=lambda x: x[0])
            return [(node_id, dist) for dist, node_id in current_nearest[:k]]

    def delete(self, node_id: int) -> bool:
        with self.lock:
            if node_id not in self.nodes:
                return False
            node = self.nodes[node_id]
            for level in range(node.level + 1):
                for neighbor_id in node.get_connections(level):
                    if neighbor_id in self.nodes:
                        self.nodes[neighbor_id].connections[level].discard(node_id)
            del self.nodes[node_id]
            if self.entry_point == node_id:
                if self.nodes:
                    self.entry_point = max(self.nodes.keys(),
                                          key=lambda x: self.nodes[x].level)
                else:
                    self.entry_point = None
            return True

    def get_stats(self) -> Dict:
        with self.lock:
            if not self.nodes:
                return {
                    "total_nodes": 0,
                    "total_searches": self.total_searches,
                    "total_insertions": self.total_insertions,
                }
            levels = [node.level for node in self.nodes.values()]
            connections_per_level = defaultdict(list)
            for node in self.nodes.values():
                for level in range(node.level + 1):
                    connections_per_level[level].append(
                        len(node.get_connections(level))
                    )
            return {
                "total_nodes": len(self.nodes),
                "max_level": max(levels),
                "avg_level": np.mean(levels),
                "entry_point_level": self.nodes[self.entry_point].level if self.entry_point else 0,
                "total_searches": self.total_searches,
                "total_insertions": self.total_insertions,
                "avg_connections_per_level": {
                    level: np.mean(conns) for level, conns in connections_per_level.items()
                },
            }
