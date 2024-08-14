
CUDA_VISIBLE_DEVICES=0 wandb agent $1 &
sleep 10
CUDA_VISIBLE_DEVICES=1 wandb agent $1 &
sleep 10
CUDA_VISIBLE_DEVICES=2 wandb agent $1 &
sleep 10
CUDA_VISIBLE_DEVICES=3 wandb agent $1 &

# kill all the python processes in bash
# sudo kill $(ps aux | grep 'python' | awk '{print $2}')

