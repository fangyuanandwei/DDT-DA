mkdir Memory1_banchmark

mkdir Memory1_banchmark/VisDA_0318_CDAN
CUDA_VISIBLE_DEVICES=3 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1.py --day 0318 --time 1023 --method CDAN_memory1 --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/VisDA_0318_CDAN/visda_s2r.txt

mkdir Memory1_banchmark/VisDA_0319_CDAN
CUDA_VISIBLE_DEVICES=3 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1.py --day 0319 --time 1023 --method CDAN_memory1 --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/VisDA_0319_CDAN/visda_s2r.txt

mkdir Memory1_banchmark/VisDA_0320_CDAN
CUDA_VISIBLE_DEVICES=3 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1.py --day 0320 --time 1023 --method CDAN_memory1 --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory1_banchmark/VisDA_0320_CDAN/visda_s2r.txt
