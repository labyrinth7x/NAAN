GPUS=1,2
export CUDA_VISIBLE_DEVICES=$GPUS

base=results
lr=0.002
wd=5e-4
alpha=0.2
clip=2
size=512
hops=120
hop1=100
hop2=10
nh=1


filename=$base/lr-$lr-wd-$wd-hops-$hop1-$hop2-$hops-batch-$size-nheads-$nh-alpha-$alpha


python3.5 train_gat2g.py \
    --batch_size $size \
    --k_hops $hop1 $hop2 \
    --nheads $nh \
    --lr $lr \
    --logs_dir $filename \
    --grad_clip $clip \
    --alpha $alpha \
    --sample_hops $hops \
