#!/bin/bash

# Check if a port number is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <port_number>"
  exit 1
fi

PORT=$1

# Find the process ID (PID) using the specified port
PID=$(lsof -t -i:$PORT)

# Check if the PID was found
if [ -z "$PID" ]; then
  echo "No process found using port $PORT."
  exit 1
fi

# Kill the process
kill -9 $PID
echo "Killed process on port $PORT (PID: $PID)."
