# Setup docker container with Neo4j image and Neo4j dump file, accessible through localhost 

## Steps

1. **Create the following directories - "backup/" and "data/", and place the docker-compose.yaml in the same directory as "backup/" and "data/"**

2. **Place dump file inside dir "backup", with file name as "neo4j.dump"**
    ```bash
    cd /projectName/backup
    ```

3. **Create .env file with the following contents and place in same dir as docker-compose.yaml, /backup and /data**
    ```bash
    # Full paths to your directories
    DATA_DIR=./data
    BACKUP_DIR=./backup

    # Port configurations (default values provided)
    BROWSER_PORT=7474
    BOLT_PORT=7687

    # Neo4j credentials
    NEO4J_USER=neo4j
    NEO4J_PASS=neo4j
    ```

4. **Run docker-compose.yaml**
    ```bash
    docker-compose up
    ```
