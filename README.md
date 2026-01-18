docker compose build

docker compose up

docker compose down


---

# build image
docker compose build

# start service with GPU support (compose.yml 'gpus: all' is honored)
docker compose up -d

# run the eval script attached to a TTY (shows tqdm)
docker compose exec -it train-eval ./docker/eval.sh
docker compose exec -it train-eval ./docker/train.sh
# or open an interactive shell first:
docker compose exec -it train-eval bash
./docker/eval.sh


docker compose ps        # list running services
docker compose logs -f   # follow logs
docker compose stop      # stop containers
docker compose down      # stop and remove containers
