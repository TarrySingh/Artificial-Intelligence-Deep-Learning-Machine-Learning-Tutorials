### ml4a-guides

These guides accompany the chapters of [ml4a.github.io](http://ml4a.github.io), providing a series of pre-baked code samples, IPython notebooks, and markdown-based tutorials guiding the interested reader in how to practically work with some of the algorithms described by ml4a.

The homepage for these is [here](http://ml4a.github.io/guides), which contains an overview of the included guides and tutorials, as well as some additional (in-progress) materials to help you get setup.

## Getting the data

Some of the notebooks use publicly available datasets. In each case, you are free to substitute it with a dataset of your choice. If you want to just go through the notebooks and use the default datasets, each one has information about where to download it from. Or, you may simply download all datasets automatically in the default location by going into the `data` folder and running the script `download.sh`, i.e.

    cd data
    ./download.sh

If you are on Mac or Windows, you may have to install unrar to run the install script.

## Running the container

You can easily run these guides using [Docker](https://www.docker.com/). With Docker installed, run the following from inside the repo:

```bash
docker build . -t ml4a
```

Once the container has successfully built, you can launch it with:

```bash
./run.sh
```

A Jupyter Notebook should now be running inside of the docker container, accessible from your host machine at `http://localhost:8888`. From Jupyter browser page, navigate to wherever you cloned `ml4a-guides` to run the notebooks. 

If that port is already occupied, you may recieve an error from the `run.sh` script. You can easily switch the port published to the docker instance like so:

```bash
JUPYTER_PORT=1337 ./run.sh # visit at http://localhost:1337 instead
```


