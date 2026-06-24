#!/bin/sh
# Ollama container entrypoint.
# Starts the server, waits until it's ready, then pulls the configured model
# only if it is not already present in the mounted volume.
# This makes subsequent container starts instant (no re-download).

set -e

MODEL="${OLLAMA_MODEL:-llama3.2}"

echo "[ollama] Starting server..."
/bin/ollama serve &
SERVE_PID=$!

# Wait for the server to accept connections (no curl needed)
echo "[ollama] Waiting for server to be ready..."
until /bin/ollama list >/dev/null 2>&1; do
    sleep 2
done
echo "[ollama] Server is ready."

# Pull model only when not already cached in the volume
if /bin/ollama list 2>/dev/null | grep -qF "${MODEL}"; then
    echo "[ollama] Model '${MODEL}' already present — skipping pull."
else
    echo "[ollama] Pulling model '${MODEL}' (first run, may take several minutes)..."
    /bin/ollama pull "${MODEL}"
    echo "[ollama] Model '${MODEL}' ready."
fi

# Hand off to the server process
wait ${SERVE_PID}
