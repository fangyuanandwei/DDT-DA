mkdir banchmark
mkdir banchmark/CLEF_CDANMemory3_banchmark
CUDA_VISIBLE_DEVICES=4 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1.py --day 0901 --time 0907 --method CDAN_memory1 --dataset CLEF --source i_split --target p_split  --batch-size 32 --max_epoch 20 | tee banchmark/CLEF_CDANMemory3_banchmark/CLEF_i2p.txt
CUDA_VISIBLE_DEVICES=4 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1.py --day 0901 --time 0907 --method CDAN_memory1 --dataset CLEF --source i_split --target c_split  --batch-size 32 --max_epoch 20 | tee banchmark/CLEF_CDANMemory3_banchmark/CLEF_i2c.txt

CUDA_VISIBLE_DEVICES=4 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1.py --day 0901 --time 0907 --method CDAN_memory1 --dataset CLEF --source p_split --target i_split  --batch-size 32 --max_epoch 20 | tee banchmark/CLEF_CDANMemory3_banchmark/CLEF_p2i.txt
CUDA_VISIBLE_DEVICES=4 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1.py --day 0901 --time 0907 --method CDAN_memory1 --dataset CLEF --source p_split --target c_split  --batch-size 32 --max_epoch 20 | tee banchmark/CLEF_CDANMemory3_banchmark/CLEF_p2c.txt

CUDA_VISIBLE_DEVICES=4 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1.py --day 0901 --time 0907 --method CDAN_memory1 --dataset CLEF --source c_split --target i_split  --batch-size 32 --max_epoch 20 | tee banchmark/CLEF_CDANMemory3_banchmark/CLEF_c2i.txt
CUDA_VISIBLE_DEVICES=4 /raid/huangl05/anaconda3.6/bin/python3.6 CDAN_memory1.py --day 0901 --time 0907 --method CDAN_memory1 --dataset CLEF --source c_split --target p_split  --batch-size 32 --max_epoch 20 | tee banchmark/CLEF_CDANMemory3_banchmark/CLEF_c2p.txt

