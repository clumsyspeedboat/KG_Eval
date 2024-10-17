# Setup docker container with Neo4j image and Neo4j dump file, accessible through localhost 

## Steps

1. **Create project directory with two subdirectories - backup and data e.g. /projectName/backup and /projectName/data:**
    ```bash
    mkdir projectName
    cd projectName
    mkdir backup
    mkdir data
    ```

2. **Place dump file inside backup, with file name as "neo4j.dump"**
    ```bash
    cd /projectName/backup
    ```

3. **Change directory (cd) to this README.md file**


4. **Run docker compose**
    ```bash
    docker-compose up
    ```