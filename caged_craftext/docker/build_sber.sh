if [ -z "$1" ]; then
    postfix=
else
    postfix=_$1
fi

cd ..

docker build -t craftext_img_sber -f docker/dockerfile_12cuda .