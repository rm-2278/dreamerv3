<!-- Run this first -->
docker build -t dreamerv3:latest .


<!-- Then run this to use GPU -->
-v will mount your local `~/dreamer_logs` directory to the container's `/app/logdir` directory.
```
docker run -it --rm --gpus all \
    -v ~/dreamer_logs:/app/logdir \
    dreamerv3:latest \
    python /app/dreamerv3/main.py \
      --logdir /app/logdir/gpu_run \
      --configs atari debug
```


-> This will only use CPU, since in debugging the config expliciltly sets platform to CPU.



```
docker run -it --rm --gpus all \
    -v ~/dreamer_logs:/app/logdir \
    dreamerv3:latest \
    python /app/dreamerv3/main.py \
      --logdir /app/logdir/gpu_run \
      --configs atari size50m
```


This gave an error for ptxas not supporting GPU capability. 

Tried changing 
```
RUN pip install jax[cuda]==0.5.0
```
to 
```
RUN pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
in Dockerfile
and 
```
nvidia-cuda-nvcc-cu12<=12.2
```
to
```
nvidia-cuda-nvcc-cu12==12.8.*
```
in requirements.txt.


Didn't support bf16, so forced to use float32 instead.
```
docker run -it --rm --gpus all \
  -e XLA_FLAGS="--xla_gpu_use_driver_ptx_compilation=true" \
  -v ~/GitHub/dreamerv3/dreamer_logs:/app/logdir \
  dreamerv3:latest \
  python /app/dreamerv3/main.py \
    --logdir /app/logdir/gpu_run \
    --configs atari size50m \
    --jax.compute_dtype float32
```

With WandB enabled (Added tp requirements.txt):
Disabled video logging to avoid extra errors.
Further monkey-patched main.py to convert from 1 to 3 channels. (line 11 to 31)

```
docker run -it --rm --gpus all \
  -e XLA_FLAGS="--xla_gpu_use_driver_ptx_compilation=true" \
  -e WANDB_API_KEY=4cd757324fc872accf3226af5faf997bdbfe08df \
  -e WANDB_DISABLE_VIDEO=1 \
  -v ~/GitHub/dreamerv3/dreamer_logs:/app/logdir \
  dreamerv3:latest \
  python /app/dreamerv3/main.py \
    --logdir /app/logdir/gpu_run \
    --configs atari size50m \
    --jax.compute_dtype float32 \
    --logger.outputs wandb
```

Got this error:

```
sudo dmesg -T | grep -i "killed"
[Sun Dec  7 07:34:43 2025] Out of memory: Killed process 397739 (python) total-vm:65353744kB, anon-rss:23669048kB, file-rss:57892kB, shmem-rss:572672kB, UID:0 pgtables:55888kB oom_score_adj:0
```

JAX and TensorFlow use shared memory for parallel data processing. Docker defaults to 64MB, which is too small. Added --shm-size=4gb (or higher, e.g., 8gb) to command as below.
(Also changed task to crafter.)

```
docker run -it --rm --gpus all \
  --shm-size=14gb \
  -e XLA_FLAGS="--xla_gpu_use_driver_ptx_compilation=true" \
  -e WANDB_API_KEY=4cd757324fc872accf3226af5faf997bdbfe08df \
  -e WANDB_DISABLE_VIDEO=1 \
  -v ~/GitHub/dreamerv3/dreamer_logs/crafter:/app/logdir \
  dreamerv3:latest \
  python /app/dreamerv3/main.py \
    --logdir /app/logdir/gpu_run \
    --configs crafter size50m \
    --jax.compute_dtype float32 \
    --logger.outputs wandb
```

TODO:
- We must check if we can use GPU seperately.
- I don't know how the agent step corresponds with the batch size and train ratio.