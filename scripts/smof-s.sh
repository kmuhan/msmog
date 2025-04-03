args="
--data ./data/wikitext-103/ \
--base_arch transformer \
--architecture sfsfsf \
--gate_name smoe \
--nlayers 3 \
--hid-sz 128 \
--inner-hid-sz 2048 \
--nheads 8 \
--block-sz 256 \
--attn-span 256 \
--dropout 0.7 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 60 \
--batch-sz 32 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint ./ckpt/smof.pt \
--full-eval-mode \
"

# echo "Training ..."
# CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train.py $args --resume --full-eval-mode
