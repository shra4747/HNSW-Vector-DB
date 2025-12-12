"""
Raft Consensus Algorithm Implementation
Provides distributed consensus for vector index replication.
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    term: int
    index: int
    command: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class RaftMessage:
    term: int
    sender_id: str


@dataclass
class RequestVoteMessage(RaftMessage):
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteResponse:
    term: int
    vote_granted: bool


@dataclass
class AppendEntriesMessage(RaftMessage):
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int


@dataclass
class AppendEntriesResponse:
    term: int
    success: bool
    match_index: int


class RaftNode:
    def __init__(
        self,
        node_id: str,
        peer_addresses: Dict[str, str],
        election_timeout_range: tuple = (150, 300),
        heartbeat_interval: int = 50,
        state_machine: Optional[Callable] = None
    ):
        self.node_id = node_id
        self.peer_addresses = peer_addresses
        self.peer_ids = list(peer_addresses.keys())
        self.election_timeout_range = election_timeout_range
        self.heartbeat_interval = heartbeat_interval / 1000.0
        self.state_machine = state_machine

        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []

        self.commit_index = 0
        self.last_applied = 0
        self.state = NodeState.FOLLOWER
        self.current_leader: Optional[str] = None

        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}

        self.last_heartbeat = time.time()
        self.election_timeout = self._get_random_timeout()

        self.stats = {
            "elections_started": 0,
            "elections_won": 0,
            "votes_received": 0,
            "heartbeats_sent": 0,
            "log_entries_replicated": 0
        }

        self._running = False
        self._tasks: List[asyncio.Task] = []
        self.http_client: Optional[httpx.AsyncClient] = None

    def _get_random_timeout(self) -> float:
        min_ms, max_ms = self.election_timeout_range
        return random.uniform(min_ms, max_ms) / 1000.0

    async def start(self):
        self._running = True
        self.http_client = httpx.AsyncClient(timeout=5.0)
        self._tasks = [
            asyncio.create_task(self._election_timer()),
            asyncio.create_task(self._heartbeat_timer()),
            asyncio.create_task(self._apply_committed_entries())
        ]
        logger.info(f"Node {self.node_id} started as {self.state.value}")

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.http_client:
            await self.http_client.aclose()
        logger.info(f"Node {self.node_id} stopped")

    async def _election_timer(self):
        while self._running:
            await asyncio.sleep(0.01)
            if self.state == NodeState.LEADER:
                continue
            time_since_heartbeat = time.time() - self.last_heartbeat
            if time_since_heartbeat >= self.election_timeout:
                await self._start_election()

    async def _heartbeat_timer(self):
        while self._running:
            await asyncio.sleep(self.heartbeat_interval)
            if self.state == NodeState.LEADER:
                await self._send_heartbeats()

    async def _start_election(self):
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat = time.time()
        self.election_timeout = self._get_random_timeout()
        self.stats["elections_started"] += 1
        logger.info(f"Node {self.node_id} starting election for term {self.current_term}")
        votes_received = 1
        votes_needed = (len(self.peer_ids) + 1) // 2 + 1
        vote_tasks = []
        for peer_id in self.peer_ids:
            vote_tasks.append(self._request_vote_from_peer(peer_id))
        vote_responses = await asyncio.gather(*vote_tasks, return_exceptions=True)
        for response in vote_responses:
            if isinstance(response, RequestVoteResponse):
                if response.vote_granted:
                    votes_received += 1
                    self.stats["votes_received"] += 1
                if response.term > self.current_term:
                    self.current_term = response.term
                    self.state = NodeState.FOLLOWER
                    self.voted_for = None
                    return
        if self.state == NodeState.CANDIDATE and votes_received >= votes_needed:
            await self._become_leader()

    async def _request_vote_from_peer(self, peer_id: str) -> Optional[RequestVoteResponse]:
        try:
            last_log_index = len(self.log)
            last_log_term = self.log[-1].term if self.log else 0
            message = {
                "term": self.current_term,
                "sender_id": self.node_id,
                "candidate_id": self.node_id,
                "last_log_index": last_log_index,
                "last_log_term": last_log_term
            }
            peer_address = self.peer_addresses[peer_id]
            response = await self.http_client.post(
                f"{peer_address}/raft/request_vote",
                json=message
            )
            if response.status_code == 200:
                data = response.json()
                return RequestVoteResponse(
                    term=data["term"],
                    vote_granted=data["vote_granted"]
                )
        except Exception as e:
            logger.debug(f"Failed to get vote from {peer_id}: {e}")
        return None

    async def _become_leader(self):
        self.state = NodeState.LEADER
        self.current_leader = self.node_id
        self.stats["elections_won"] += 1
        last_log_index = len(self.log)
        for peer_id in self.peer_ids:
            self.next_index[peer_id] = last_log_index + 1
            self.match_index[peer_id] = 0
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
        await self._send_heartbeats()

    async def _send_heartbeats(self):
        if self.state != NodeState.LEADER:
            return
        self.stats["heartbeats_sent"] += 1
        heartbeat_tasks = []
        for peer_id in self.peer_ids:
            heartbeat_tasks.append(self._send_append_entries(peer_id))
        responses = await asyncio.gather(*heartbeat_tasks, return_exceptions=True)
        for peer_id, response in zip(self.peer_ids, responses):
            if isinstance(response, AppendEntriesResponse):
                if response.success:
                    self.match_index[peer_id] = response.match_index
                    self.next_index[peer_id] = response.match_index + 1
                if response.term > self.current_term:
                    self.current_term = response.term
                    self.state = NodeState.FOLLOWER
                    self.voted_for = None
                    self.current_leader = None
                    return
        self._update_commit_index()

    async def _send_append_entries(self, peer_id: str) -> Optional[AppendEntriesResponse]:
        try:
            next_index = self.next_index.get(peer_id, 1)
            prev_log_index = next_index - 1
            prev_log_term = self.log[prev_log_index - 1].term if prev_log_index > 0 and prev_log_index <= len(self.log) else 0
            entries = []
            if next_index <= len(self.log):
                entries = self.log[next_index - 1:]
            message = {
                "term": self.current_term,
                "sender_id": self.node_id,
                "leader_id": self.node_id,
                "prev_log_index": prev_log_index,
                "prev_log_term": prev_log_term,
                "entries": [{"term": e.term, "index": e.index, "command": e.command, "timestamp": e.timestamp} for e in entries],
                "leader_commit": self.commit_index
            }
            peer_address = self.peer_addresses[peer_id]
            response = await self.http_client.post(
                f"{peer_address}/raft/append_entries",
                json=message
            )
            if response.status_code == 200:
                data = response.json()
                return AppendEntriesResponse(
                    term=data["term"],
                    success=data["success"],
                    match_index=data["match_index"]
                )
        except Exception as e:
            logger.debug(f"Failed to send append_entries to {peer_id}: {e}")
        return None

    def _update_commit_index(self):
        if self.state != NodeState.LEADER:
            return
        match_indices = [self.match_index.get(peer_id, 0) for peer_id in self.peer_ids]
        match_indices.append(len(self.log))
        match_indices.sort(reverse=True)
        majority_index = match_indices[len(match_indices) // 2]
        if majority_index > self.commit_index:
            if majority_index <= len(self.log) and self.log[majority_index - 1].term == self.current_term:
                self.commit_index = majority_index
                logger.debug(f"Leader {self.node_id} updated commit_index to {self.commit_index}")

    async def append_entry(self, command: Dict[str, Any], timeout: float = 5.0) -> bool:
        if self.state != NodeState.LEADER:
            return False
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log) + 1,
            command=command
        )
        self.log.append(entry)
        self.stats["log_entries_replicated"] += 1
        logger.debug(f"Leader {self.node_id} appended entry {entry.index}")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.commit_index >= entry.index:
                return True
            await asyncio.sleep(0.01)
        return self.commit_index >= entry.index

    async def _apply_committed_entries(self):
        while self._running:
            await asyncio.sleep(0.01)
            while self.last_applied < self.commit_index:
                self.last_applied += 1
                entry = self.log[self.last_applied - 1]
                if self.state_machine:
                    try:
                        await self.state_machine(entry.command)
                        logger.debug(f"Node {self.node_id} applied entry {entry.index}")
                    except Exception as e:
                        logger.error(f"Failed to apply entry {entry.index}: {e}")

    def handle_request_vote(self, msg: RequestVoteMessage) -> RequestVoteResponse:
        vote_granted = False
        if msg.term > self.current_term:
            self.current_term = msg.term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        if msg.term == self.current_term:
            if self.voted_for is None or self.voted_for == msg.candidate_id:
                last_log_index = len(self.log)
                last_log_term = self.log[-1].term if self.log else 0
                log_ok = (msg.last_log_term > last_log_term or
                         (msg.last_log_term == last_log_term and
                          msg.last_log_index >= last_log_index))
                if log_ok:
                    vote_granted = True
                    self.voted_for = msg.candidate_id
                    self.last_heartbeat = time.time()
        return RequestVoteResponse(term=self.current_term, vote_granted=vote_granted)

    def handle_append_entries(self, msg: AppendEntriesMessage) -> AppendEntriesResponse:
        success = False
        if msg.term > self.current_term:
            self.current_term = msg.term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
        if msg.term == self.current_term:
            self.state = NodeState.FOLLOWER
            self.current_leader = msg.leader_id
            self.last_heartbeat = time.time()
            if msg.prev_log_index == 0 or (
                msg.prev_log_index <= len(self.log) and
                self.log[msg.prev_log_index - 1].term == msg.prev_log_term
            ):
                success = True
                if msg.entries:
                    if msg.prev_log_index < len(self.log):
                        self.log = self.log[:msg.prev_log_index]
                    for entry in msg.entries:
                        self.log.append(entry)
                if msg.leader_commit > self.commit_index:
                    self.commit_index = min(msg.leader_commit, len(self.log))
        match_index = len(self.log)
        return AppendEntriesResponse(
            term=self.current_term,
            success=success,
            match_index=match_index
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "term": self.current_term,
            "log_length": len(self.log),
            "commit_index": self.commit_index,
            "is_leader": self.state == NodeState.LEADER,
            "stats": self.stats.copy()
        }
