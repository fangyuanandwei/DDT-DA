#office31
#BCDM
mkdir Memory1_banchmark
mkdir Memory1_banchmark/office31_0731/

CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0731 --time 1051 --method DANN_memory1 --dataset office31 --source amazon --target dslr  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/office31_0731/office31_DANN_a2d.txt
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0731 --time 1051 --method DANN_memory1 --dataset office31 --source amazon --target webcam  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/office31_0731/office31_DANN_a2w.txt

CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0731 --time 1051 --method DANN_memory1 --dataset office31 --source dslr --target amazon  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/office31_0731/office31_DANN_d2a.txt
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0731 --time 1051 --method DANN_memory1 --dataset office31 --source dslr --target webcam  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/office31_0731/office31_DANN_d2w.txt

CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0731 --time 1051 --method DANN_memory1 --dataset office31 --source webcam --target amazon  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/office31_0731/office31_DANN_w2a.txt
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0731 --time 1051 --method DANN_memory1 --dataset office31 --source webcam --target dslr  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/office31_0731/office31_DANN_w2d.txt