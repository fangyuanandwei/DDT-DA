mkdir banchmark3
mkdir banchmark3/Office31_CDANMemory3_banchmark
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3.py --day 0831 --time 2041 --method CDAN_memory1 --dataset office31 --source amazon --target dslr  --batch-size 32 --max_epoch 20 | tee banchmark3/Office31_CDANMemory3_banchmark/office31_a2d.txt
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3.py --day 0831 --time 2041 --method CDAN_memory1 --dataset office31 --source amazon --target webcam  --batch-size 32 --max_epoch 20 | tee banchmark3/Office31_CDANMemory3_banchmark/office31_a2w.txt

CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3.py --day 0831 --time 2041 --method CDAN_memory1 --dataset office31 --source dslr --target amazon  --batch-size 32 --max_epoch 20 | tee banchmark3/Office31_CDANMemory3_banchmark/office31_d2a.txt
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3.py --day 0831 --time 2041 --method CDAN_memory1 --dataset office31 --source dslr --target webcam  --batch-size 32 --max_epoch 20 | tee banchmark3/Office31_CDANMemory3_banchmark/office31_d2w.txt

CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3.py --day 0831 --time 2041 --method CDAN_memory1 --dataset office31 --source webcam --target amazon  --batch-size 32 --max_epoch 20 | tee banchmark3/Office31_CDANMemory3_banchmark/office31_w2a.txt
CUDA_VISIBLE_DEVICES=6 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory3.py --day 0831 --time 2041 --method CDAN_memory1 --dataset office31 --source webcam --target dslr  --batch-size 32 --max_epoch 20 | tee banchmark3/Office31_CDANMemory3_banchmark/office31_w2d.txt

