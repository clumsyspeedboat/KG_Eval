# Setup docker container with Neo4j image and Neo4j dump file, accessible through localhost 

## Steps


1. **Place dump file inside dir "backup", with file name as "neo4j.dump"**
    ```bash
    cd /projectName/backup
    ```

2. **Run docker-compose.yaml**
    ```bash
    docker-compose up
    ```