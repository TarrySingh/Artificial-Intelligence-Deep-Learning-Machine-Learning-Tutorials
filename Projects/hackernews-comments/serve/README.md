# Serving

## TensorFlow Serving
We can use a TensorFlow docker image to serve the model, so that it replies to GRPC (adjust path and model name as needed).
```
docker run \
  --rm \
  -p 9000:9000 \
  -v $PWD:/models \
  --name serving \
  --entrypoint tensorflow_model_server \
  tensorflow/serving:1.11.0 --enable_batching=true \
                            --batching_parameters_file=/models/batching_parameters.txt \
                            --port=9000 --model_base_path=/models/export1/1550276061 \
                            --model_name=hncynic
```
Or on GPU:
```
nvidia-docker run \
  -d \
  --rm \
  -p 9000:9000 \
  -v $PWD:/models \
  --name serving \
  --entrypoint tensorflow_model_server \
  tensorflow/serving:1.11.0-gpu --enable_batching=true \
                                --batching_parameters_file=/models/batching_parameters.txt \
                                --port=9000 --model_base_path=/models/export1/1550276061 \
                                --model_name=hncynic
```

## Querying
Once the docker container is running, `client.py` can be used to query the model (adjust paths as needed):
```
echo Why I Hate Whiteboard Interviews \
  | ./client.py --host=localhost --port=9000 --model_name=hncynic \
                --preprocessor=../data/preprocess.sh \
                --bpe_codes=../exps/data/bpecodes \
                --postprocessor=../data/mosesdecoder/scripts/tokenizer/detokenizer.perl
```
Output is printed in two columns (tab-separated), where the first column is the title and the second a sampled comment.
