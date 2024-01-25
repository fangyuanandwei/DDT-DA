mkdir Memory3_banchmark_part
mkdir Memory3_banchmark_part/0318_Office31_CDANMemory3_Part
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3_part.py --day 0318 --time 1200 --method CDAN_memory3 --dataset office31 --source amazon --target dslr  --batch-size 32 --max_epoch 20 | tee Memory3_banchmark_part/0318_Office31_CDANMemory3_Part/office31_a2d.txt
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3_part.py --day 0318 --time 1200 --method CDAN_memory3 --dataset office31 --source amazon --target webcam  --batch-size 32 --max_epoch 20 | tee Memory3_banchmark_part/0318_Office31_CDANMemory3_Part/office31_a2w.txt

CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3_part.py --day 0318 --time 1200 --method CDAN_memory3 --dataset office31 --source dslr --target amazon  --batch-size 32 --max_epoch 20 | tee Memory3_banchmark_part/0318_Office31_CDANMemory3_Part/office31_d2a.txt
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3_part.py --day 0318 --time 1200 --method CDAN_memory3 --dataset office31 --source dslr --target webcam  --batch-size 32 --max_epoch 20 | tee Memory3_banchmark_part/0318_Office31_CDANMemory3_Part/office31_d2w.txt

CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3_part.py --day 0318 --time 1200 --method CDAN_memory3 --dataset office31 --source webcam --target amazon  --batch-size 32 --max_epoch 20 | tee Memory3_banchmark_part/0318_Office31_CDANMemory3_Part/office31_w2a.txt
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3_part.py --day 0318 --time 1200 --method CDAN_memory3 --dataset office31 --source webcam --target dslr  --batch-size 32 --max_epoch 20 | tee Memory3_banchmark_part/0318_Office31_CDANMemory3_Part/office31_w2d.txt

