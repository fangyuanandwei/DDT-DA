mkdir Memory1_banchmark
mkdir Memory1_banchmark/VisDA_0317_CDAN
CUDA_VISIBLE_DEVICES=4 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1.py --day 0317 --time 1509 --method CDAN_memory1 --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/VisDA_0317_CDAN/visda_s2r.txt
