<!-- Run this first -->
docker build -t dreamerv3:latest .


<!-- Then run this to use GPU -->
-v will mount your local `~/dreamer_logs` directory to the container's `/app/logdir` directory.
'''
docker run -it --rm --gpus all \
    -v ~/dreamer_logs:/app/logdir \
    dreamerv3:latest \
    python /app/dreamerv3/main.py \
      --logdir /app/logdir/gpu_run \
      --configs atari
'''


-> This will only use CPU, since in debugging the config expliciltly sets platform to CPU.

TODO:
- We must check if we can use GPU seperately.
- I don't know how the agent step corresponds with the batch size and train ratio.