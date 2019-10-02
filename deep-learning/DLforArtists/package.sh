#!/usr/bin/env bash

#adapted from https://github.com/kylemcdonald/ml-notebook/blob/master/package.sh

IMAGE="ml4a"
IMAGE_FILE="ml4a.tar"

if [ ! -e $IMAGE_FILE ] ; then
	echo "Saving $IMAGE to $IMAGE_FILE"
	docker save $IMAGE > $IMAGE_FILE
fi
echo "$IMAGE_FILE is ready, zipping everything..."

rm shared/jupyter.log

HASH=`git rev-parse HEAD | cut -c-8`
ZIPFILE="../ml4a-guides-$HASH.zip"

zip -q -r "$ZIPFILE" ./

echo "Package is ready: $ZIPFILE"