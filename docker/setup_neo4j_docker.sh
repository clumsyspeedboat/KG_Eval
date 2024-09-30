#!/bin/bash

# Prompt the user for the directories
read -p "Enter the full path for the Neo4j data directory (e.g., /Users/kennymccormick/Desktop/Neo4j/ckg_rag/data): " DATA_DIR
read -p "Enter the full path for the Neo4j backup directory (e.g., /Users/kennymccormick/Desktop/Neo4j/ckg_rag/backup): " BACKUP_DIR

# Prompt the user for the Neo4j Browser (HTTP) port, explaining what it's for
read -p "Enter the port number for the Neo4j Browser (default 7474, which is accessed via http://localhost:7474): " BROWSER_PORT
BROWSER_PORT=${BROWSER_PORT:-7474}  # Default to 7474 if not provided

# Prompt the user for the Neo4j Bolt protocol port, explaining what it's for
read -p "Enter the port number for the Neo4j Bolt protocol (default 7687, used for database access via drivers): " BOLT_PORT
BOLT_PORT=${BOLT_PORT:-7687}  # Default to 7687 if not provided

# Prompt the user for the username (default is "neo4j")
read -p "Enter the Neo4j username (default 'neo4j'): " NEO4J_USER
NEO4J_USER=${NEO4J_USER:-neo4j}  # Default to 'neo4j' if not provided

# Prompt the user for the password (default is "password")
read -p "Enter the Neo4j password (default 'password'): " NEO4J_PASS
NEO4J_PASS=${NEO4J_PASS:-password}  # Default to 'password' if not provided

# Confirm the directories, ports, and credentials entered by the user
echo "Data directory: $DATA_DIR"
echo "Backup directory: $BACKUP_DIR"
echo "Neo4j Browser will be accessible on http://localhost:$BROWSER_PORT"
echo "Bolt protocol will be accessible on port $BOLT_PORT"
echo "Username: $NEO4J_USER"
echo "Password: $NEO4J_PASS"
read -p "Is this correct? (y/n): " CONFIRMATION

if [[ $CONFIRMATION != "y" ]]; then
  echo "Aborting script. Please run again and enter the correct values."
  exit 1
fi

# Step 1: Load the backup into the Neo4j database
echo "Loading the backup into the Neo4j database..."
docker run --interactive --tty --rm \
  --volume=$DATA_DIR:/data \
  --volume=$BACKUP_DIR:/backup \
  neo4j neo4j-admin database load neo4j --from-path=/backup --overwrite-destination=true

# Check if the backup load was successful
if [ $? -eq 0 ]; then
  echo "Backup loaded successfully."

  # Step 2: Start and expose the Neo4j instance
  echo "Starting the Neo4j instance and exposing it on ports $BROWSER_PORT (Browser) and $BOLT_PORT (Bolt)..."
  docker run --interactive --tty --rm \
    --volume=$DATA_DIR:/data \
    --publish=$BROWSER_PORT:7474 \
    --publish=$BOLT_PORT:7687 \
    --env NEO4J_AUTH=$NEO4J_USER/$NEO4J_PASS \
    neo4j:5.23.0

else
  echo "Failed to load the backup. Aborting Neo4j start."
  exit 1
fi
