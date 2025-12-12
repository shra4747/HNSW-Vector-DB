#!/bin/bash
# Start dynamic Raft clusters for distributed vector database

# Default values
NUM_NODES=3
RESET_DATA=false
START_PORT=8001

# Parse command line arguments
show_usage() {
    echo "Usage: $0 [num_nodes] [--reset]"
    echo ""
    echo "Arguments:"
    echo "  num_nodes    Number of nodes to start (default: 3, must be odd for Raft)"
    echo "  --reset      Clear all existing data before starting"
    echo ""
    echo "Examples:"
    echo "  $0           # Start 3 nodes"
    echo "  $0 5         # Start 5 nodes"
    echo "  $0 3 --reset # Start 3 nodes with fresh data"
    echo "  $0 --reset   # Start 3 nodes with fresh data"
    exit 1
}

for arg in "$@"; do
    case $arg in
        --reset)
            RESET_DATA=true
            ;;
        --help|-h)
            show_usage
            ;;
        [0-9]*)
            NUM_NODES=$arg
            ;;
        *)
            echo "Unknown argument: $arg"
            show_usage
            ;;
    esac
done

if [ $((NUM_NODES % 2)) -eq 0 ]; then
    echo "Number of nodes should be odd for Raft consensus."
    echo "   Even number of nodes doesn't improve fault tolerance."
    echo "   Consider using $((NUM_NODES + 1)) nodes instead."
    echo ""
fi

if [ $NUM_NODES -lt 3 ]; then
    echo "Minimum 3 nodes required for Raft consensus"
    exit 1
fi

echo "Starting $NUM_NODES-node Raft cluster..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

PORTS=""
for i in $(seq 0 $((NUM_NODES - 1))); do
    PORT=$((START_PORT + i))
    if [ -z "$PORTS" ]; then
        PORTS="$PORT"
    else
        PORTS="$PORTS,$PORT"
    fi
done

echo "Killing any existing processes on ports $PORTS..."
lsof -ti:$PORTS | xargs kill -9 2>/dev/null || true

# Clear data directories if --reset flag is set
if [ "$RESET_DATA" = true ]; then
    echo "Clearing old data directories..."
    for i in $(seq 1 $NUM_NODES); do
        rm -rf "./data/node$i"
    done
    echo "✓ Data cleared"
fi

build_peer_list() {
    local node_num=$1
    local peers=""

    for i in $(seq 1 $NUM_NODES); do
        if [ $i -ne $node_num ]; then
            local peer_port=$((START_PORT + i - 1))
            if [ -z "$peers" ]; then
                peers="node$i:$peer_port"
            else
                peers="$peers,node$i:$peer_port"
            fi
        fi
    done

    echo "$peers"
}

PIDS=()
echo ""
for i in $(seq 1 $NUM_NODES); do
    NODE_ID="node$i"
    NODE_PORT=$((START_PORT + i - 1))
    PEERS=$(build_peer_list $i)

    NODE_ID=$NODE_ID NODE_PORT=$NODE_PORT PEERS="$PEERS" python api/rest/distributed_server.py &
    PID=$!
    PIDS+=($PID)
    echo "Started $NODE_ID on port $NODE_PORT (PID: $PID)"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Cluster started successfully!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Nodes: $NUM_NODES"
echo "  - Port range: $START_PORT-$((START_PORT + NUM_NODES - 1))"
echo "  - Data reset: $RESET_DATA"
echo ""
echo "API endpoints:"
for i in $(seq 1 $NUM_NODES); do
    PORT=$((START_PORT + i - 1))
    echo "  - Node $i: http://localhost:$PORT"
done
echo ""
echo "Quick commands:"
echo "  Check health:  curl http://localhost:$START_PORT/health"
echo "  View stats:    curl http://localhost:$START_PORT/stats"
echo "  Run tests:     python test_cluster.py"
echo ""
echo "Press Ctrl+C to stop all nodes..."
echo "═══════════════════════════════════════════════════════════"

cleanup() {
    echo ""
    echo "Stopping cluster..."
    for PID in "${PIDS[@]}"; do
        kill $PID 2>/dev/null
    done
    echo "Cluster stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM
wait
