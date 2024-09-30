# Setup docker container with Neo4j image and Neo4j dump file, accessible through localhost 

## Steps

1. **Create project directory with two subdirectories - backup and data e.g. /projectName/backup and /projectName/data:**
    ```bash
    mkdir projectName
    cd projectName
    mkdir backup
    mkdir data
    ```

2. **Place dump file inside backup, for e.g. if dump file name is "neo4j.dump" then you should be able to the file when you run:**
    ```bash
    cd /projectName/backup
    ```

3. **Change directory (cd) where you placed the shell script "setup_neo4j_docker.sh" (preferrable inside "projectName" along with "bakup" and "data"):**


4. **Make script "setup_neo4j_docker.sh" executable:**
    ```bash
    chmod +x setup_neo4j_docker.sh
    ```

5. **Run script (Provide user input as asked by the script):**
    ```bash
    bash setup_neo4j_docker.sh
    ```