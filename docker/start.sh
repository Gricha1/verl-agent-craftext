
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

docker run -it --rm --name $container_name --memory="200g" --shm-size=8g --gpus '"device=0,1"' --env WANDB_API_KEY=$WANDB_API_KEY -v $(pwd):/usr/home/workspace -v $(pwd)/logdir:/root/logdir $image_name
