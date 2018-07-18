#!/usr/bin/env bash

#adapted from https://github.com/kylemcdonald/ml-notebook/blob/master/run.sh

IMAGE="ml4a"
IMAGE_FILE="$IMAGE.tar"
JUPYTER_PORT=${JUPYTER_PORT:-8888}
HOST_IP=0.0.0.0
DIR=`pwd`

if ! ( docker images | grep "$IMAGE" &>/dev/null ) ; then
	if [ -e $IMAGE_FILE ]; then
		echo "The image will be loaded from $IMAGE_FILE (first time only, ~1 minute)."
		docker load < $IMAGE_FILE
	else
		echo "The image will be downloaded from docker (first time only)."
	fi
fi

docker run --publish $JUPYTER_PORT:$JUPYTER_PORT --volume=$DIR/:$DIR/ -ti $IMAGE /bin/bash -c "ln /dev/null /dev/raw1394 ; jupyter notebook --allow-root --no-browser --ip=$HOST_IP --port=$JUPYTER_PORT --notebook-dir=$DIR"