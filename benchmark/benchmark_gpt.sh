#! /bin/bash
GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=29011
NNODES=1
NODE_RANK=0
TP_SIZE=4
PP_SIZE=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=/home/lcjzy/data2/workspace/GPT/ckpts
VOCAB_FILE=/data/scratch/Megatron_data/vocab/gpt2-vocab.json
MERGE_FILE=/data/scratch/Megatron_data/vocab/gpt2-merges.txt
DATA_PATH=/home/lcjzy/data2/workspace/GPT/data/my-gpt2_text_document
GPT_ARGS="--num-layers 50 --hidden-size 4096 --num-attention-heads 16 --seq-length 1024 --max-position-embeddings 1024 --micro-batch-size 8 --lr 0.0005 --train-iters 150000 --lr-decay-iters 150000 --lr-decay-style cosine --lr-warmup-iters 1 --weight-decay .1 --adam-beta2 .999 --fp16 --log-interval 10 --save-interval 2000 --eval-interval 200 --eval-iters 10"
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
        benchmark_gpt.py \
        --tensor-model-parallel-size $TP_SIZE \
        --pipeline-model-parallel-size $PP_SIZE \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS \
        --use-distributed-optimizer \
        --recompute-granularity 'full' \
        --recompute-method 'uniform'