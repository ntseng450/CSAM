  #!/bin/bash

docker build \
    --network host \
    -t csam:1.0.0 \
    ./docker