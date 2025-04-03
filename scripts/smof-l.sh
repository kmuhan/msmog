args="
--data ./data/wikitext-103/ \
--base_arch transformer \
--architecture sfsfsfsfsfsfsfsfsfsfsfsf \
--gate_name smoe \
--nlayers 12 \
--hid-sz 512 \
--inner-hid-sz 8192 \
--nheads 8 \
--block-sz 1024 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 5000 \
--niter 80 \
--batch-sz 4 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint ./ckpt/smof-l.pt \
--full-eval-mode \
"

# echo "Training ..."
# CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train.py $args --resume --full-eval-mode
