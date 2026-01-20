# Running Training and Evaluation with Docker Compose

This project uses **Docker Compose** to ensure a reproducible runtime environment with GPU support. Training and evaluation are executed inside a container, while configuration is controlled via YAML files.

---

Configuration Files
- `train.yaml` Defines all training-related parameters, including:
    - model architecture
    - dataset paths
    - hyperparameters
    - optimizer and scheduler settings

- `eval.yaml` Defines evaluation-specific parameters, including:
    - checkpoint paths
    - evaluation metrics
    - dataset splits and output directories

Modify these files before running training or evaluation to adjust the experiment setup.

## 1. Build the Docker Image
Build the Docker image defined in `docker-compose.yml`:

```
docker compose build
```

## 2. Start the Services (GPU Enabled)
Start the container in detached mode. GPU support is enabled via `gpus: all` in `docker-compose.yml`.

```
docker compose up -d
```

Verify that the service is running:

```
docker compose ps
```

## 3. Run Training
Training is executed via the provided shell script inside the running container. The script reads its configuration from `train.yaml`.

```
docker compose exec -it train-eval ./docker/train.sh
```

Progress bars (e.g., `tqdm`) are displayed because the command is attached to a TTY.

## 4. Run Evaluation
Evaluation is executed in the same container and uses `eval.yaml` for configuration.

```
docker compose exec -it train-eval ./docker/eval.sh
```

## 5. Container Management
Stop running containers (without removing them):

```
docker compose stop
```

Stop and remove containers and networks:

```
docker compose down
```