mkdir Memory1_banchmark_part
mkdir Memory1_banchmark_part/0318_Office31_CDANMemory1_banchmark
CUDA_VISIBLE_DEVICES=2 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1_part.py --day 0318 --time 1200 --method CDAN_memory1 --dataset office31 --source amazon --target dslr  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark_part/0318_Office31_CDANMemory1_banchmark/office31_a2d.txt
CUDA_VISIBLE_DEVICES=2 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1_part.py --day 0318 --time 1200 --method CDAN_memory1 --dataset office31 --source amazon --target webcam  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark_part/0318_Office31_CDANMemory1_banchmark/office31_a2w.txt

CUDA_VISIBLE_DEVICES=2 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1_part.py --day 0318 --time 1200 --method CDAN_memory1 --dataset office31 --source dslr --target amazon  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark_part/0318_Office31_CDANMemory1_banchmark/office31_d2a.txt
CUDA_VISIBLE_DEVICES=2 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1_part.py --day 0318 --time 1200 --method CDAN_memory1 --dataset office31 --source dslr --target webcam  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark_part/0318_Office31_CDANMemory1_banchmark/office31_d2w.txt

CUDA_VISIBLE_DEVICES=2 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1_part.py --day 0318 --time 1200 --method CDAN_memory1 --dataset office31 --source webcam --target amazon  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark_part/0318_Office31_CDANMemory1_banchmark/office31_w2a.txt
CUDA_VISIBLE_DEVICES=2 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1_part.py --day 0318 --time 1200 --method CDAN_memory1 --dataset office31 --source webcam --target dslr  --batch-size 32 --max_epoch 20 | tee Memory1_banchmark_part/0318_Office31_CDANMemory1_banchmark/office31_w2d.txt

