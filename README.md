docker compose build

docker compose up

docker compose down


---

# build image
docker compose build

# start service with GPU support (compose.yml 'gpus: all' is honored)
docker compose up -d

# run the eval script attached to a TTY (shows tqdm)
docker compose exec -it rcan-train ./docker/eval.sh
# or open an interactive shell first:
docker compose exec -it rcan-train bash
./docker/eval.sh