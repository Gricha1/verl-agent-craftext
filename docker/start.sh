
if [ -z "$1" ]; then
    gpus=0
else
    gpus=$1
fi


if [ -z "$2" ]; then
    container_postfix=
else
    container_postfix=$2
fi


image_name=safe_llm_img
container_name=safe_llm_$container_postfix

echo image name --- $image_name
cd ..

if [ -d "logdir" ]; then
    echo container name --- $container_name
else
    mkdir logdir
    echo create log dir 
    echo container name --- $container_name
fi
echo gpus in docker --- $gpus

# Repo is a volume: edit caged_craftext on host without rebuilding the image.
# First start runs setup_caged_craftext_deps.sh (~15 min); then .docker_caged_deps_installed skips it.
# After caged_craftext changes: bash docker/setup_caged_craftext_editable.sh
# SKIP_CAGED_CRAFTEXT_SETUP=1 — skip even first-time install (image already has deps)
# FORCE_CAGED_CRAFTEXT_SETUP=1 — rerun full install
docker run -it --rm --name $container_name --memory="200g" --shm-size=8g --gpus '"device=0,1"' \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --env SKIP_CAGED_CRAFTEXT_SETUP="${SKIP_CAGED_CRAFTEXT_SETUP:-}" \
  --env FORCE_CAGED_CRAFTEXT_SETUP="${FORCE_CAGED_CRAFTEXT_SETUP:-}" \
  -v "$(pwd)":/usr/home/workspace \
  -v "$(pwd)/logdir":/root/logdir \
  $image_name
