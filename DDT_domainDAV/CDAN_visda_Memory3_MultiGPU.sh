mkdir Memory3_banchmark_MultiGPU
mkdir Memory3_banchmark_MultiGPU/Visda_0321_CDAN
CUDA_VISIBLE_DEVICES=1,2 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3_MultiGPU.py --day 0321 --time 1200 --method CDAN_memory3_MultiGPU --dataset VisDA --source synthetic --target real --batch-size 32 --max_epoch 20 | tee Memory3_banchmark_MultiGPU/Visda_0321_CDAN/visda_s2r.txt
