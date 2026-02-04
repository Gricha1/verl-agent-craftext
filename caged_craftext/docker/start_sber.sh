if [ -z "$1" ]; then
    device=0
else
    device=$1
fi

if [ -z "$2" ]; then
    docker_container_idx=0
else
    docker_container_idx=$2
fi

if [ -z "$3" ]; then
    image_name=craftext_img_sber
else
    image_name=$3
fi

echo "start dockergpu device: $device"
echo "start docker name: ggorbov.craftext_$docker_container_idx"
echo "start docker image: $image_name"

cd ..

# Prepare mount arguments
mount_args=""
for item in *; do
    if [ "$item" == "Craftax" ]; then
        continue
    fi
    abs_path="$(pwd)/$item"
    mount_args="$mount_args -v $abs_path:/usr/home/workspace/$item"
done


docker run -it --rm --name ggorbov.craftext_$docker_container_idx \
           --gpus "device=$device" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
           $mount_args \
           $image_name "bash"