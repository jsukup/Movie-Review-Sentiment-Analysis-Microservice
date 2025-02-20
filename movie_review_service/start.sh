#!/bin/bash
set -e

echo "Starting initialization process..."

# Function to test database connection
wait_for_db() {
    echo "Waiting for database to be ready..."
    while ! pg_isready -h db -p 5432 -U postgres > /dev/null 2>&1; do
        echo "Database is unavailable - sleeping"
        sleep 1
    done
    echo "Database is up and ready!"
}

# Install postgresql-client for pg_isready
apt-get update && apt-get install -y postgresql-client

# Wait for database to be ready
wait_for_db

# Ensure we're in the correct directory
cd /app

# Run initialization if not already done
if [ ! -f "/app/.initialized" ]; then
    echo "Running first-time initialization..."
    python init_db.py
    touch /app/.initialized
else
    echo "Running migrations upgrade..."
    aerich upgrade
fi

# Start the application
echo "Starting FastAPI application..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload