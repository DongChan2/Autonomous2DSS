#! /bin/bash
docker build -t 2dsegmentation docker/
docker run --gpus all --shm-size=8g -it -v ${PWD}:/workspace --name 2dsegmentation -w /workspace 2dsegmentation \ bash -c "pip install -r requirements.txt && \
                                                                                                                    python ./data/changename.py&& \
                                                                                                                    python ./data/preprocess.py && \
                                                                                                                    exec bash"