#!/bin/sh
# Container entrypoint. Starts the model server and keeps it alive: if it ever
# crashes, it comes back and reloads the saved index from /app/index.
# This loop never exits, which is also what keeps the container running.
mkdir -p /app/output /app/index /app/logs
cd /app
while true; do
    python3 /app/server.py >>/app/logs/server.log 2>&1
    echo "$(date -u +%T) server exited with $?, restarting" >>/app/logs/server.log
    sleep 2
done
