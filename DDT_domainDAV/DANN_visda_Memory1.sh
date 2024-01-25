#VisDA
#DANN_memory1
mkdir Memory1_banchmark

mkdir Memory1_banchmark/VisDA_0318_DANN/
CUDA_VISIBLE_DEVICES=1 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0318 --time 0001 --method DANN_memory1 --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/VisDA_0318_DANN/VisDA_DANN.txt

mkdir Memory1_banchmark/VisDA_0319_DANN/
CUDA_VISIBLE_DEVICES=1 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0319 --time 0001 --method DANN_memory1 --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/VisDA_0319_DANN/VisDA_DANN.txt

mkdir Memory1_banchmark/VisDA_0320_DANN/
CUDA_VISIBLE_DEVICES=1 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0320 --time 0001 --method DANN_memory1 --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/VisDA_0320_DANN/VisDA_DANN.txt