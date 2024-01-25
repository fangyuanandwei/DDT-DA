mkdir Memory3_banchmark

mkdir Memory3_banchmark/Visda_0318_CDAN
CUDA_VISIBLE_DEVICES=2 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3.py --day 0318 --time 0001 --method CDAN_memory3 --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/Visda_0318_CDAN/visda_s2r.txt

mkdir Memory3_banchmark/Visda_0319_CDAN
CUDA_VISIBLE_DEVICES=2 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3.py --day 0319 --time 0001 --method CDAN_memory3 --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/Visda_0319_CDAN/visda_s2r.txt

mkdir Memory3_banchmark/Visda_0320_CDAN
CUDA_VISIBLE_DEVICES=2 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3.py --day 0320 --time 0001 --method CDAN_memory3 --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/Visda_0320_CDAN/visda_s2r.txt
