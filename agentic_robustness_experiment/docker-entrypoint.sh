#!/usr/bin/env bash
set -e

# Detect host path for our working directory via the mounted Docker socket
CONTAINER_PATH="/workspace/agentic-robustness-experiment"

if [ -S /var/run/docker.sock ] && command -v docker >/dev/null 2>&1; then
    # Find the current container's ID
    CID=$(cat /proc/self/cgroup | grep -oP '(?<=docker[-/])[a-f0-9]{64}' | head -1)
    # Fallback for cgroup v2
    if [ -z "$CID" ]; then
        CID=$(cat /proc/self/mountinfo | grep -oP '(?<=docker/containers/)[a-f0-9]{64}' | head -1)
    fi

    if [ -n "$CID" ]; then
        HOST_PATH=$(docker inspect "$CID" \
            --format "{{range .Mounts}}{{if eq .Destination \"$CONTAINER_PATH\"}}{{.Source}}{{end}}{{end}}" 2>/dev/null || true)

        if [ -n "$HOST_PATH" ]; then
            export HOST_WORKSPACE_PREFIX="$HOST_PATH"
            export CONTAINER_WORKSPACE_PREFIX="$CONTAINER_PATH"
            echo "[entrypoint] Host path: $HOST_PATH"
            echo "[entrypoint] Container path: $CONTAINER_PATH"
            # Persist for future shells
            echo "export HOST_WORKSPACE_PREFIX=\"$HOST_PATH\"" >> /etc/profile.d/workspace-paths.sh
            echo "export CONTAINER_WORKSPACE_PREFIX=\"$CONTAINER_PATH\"" >> /etc/profile.d/workspace-paths.sh
        fi
    fi
fi

# Hand off to whatever command was passed (or bash by default)
exec "$@"