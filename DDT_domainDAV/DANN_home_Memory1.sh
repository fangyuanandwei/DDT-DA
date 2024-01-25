#officehome
#DANN_memory1
mkdir Officehome_Memory1_banchmark_0729
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source Art --target Clipart  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_A2C.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source Art --target Product  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_A2P.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source Art --target RealWorld  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_A2R.txt

CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source Clipart --target Art  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_C2A.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source Clipart --target Product  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_C2P.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source Clipart --target RealWorld  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_C2R.txt

CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source Product --target Art  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_P2A.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source Product --target Clipart  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_P2C.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source Product --target RealWorld  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_P2R.txt

CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source RealWorld --target Art  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_R2A.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source RealWorld --target Clipart  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_R2C.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory1.py --day 0729 --time 1051 --method DANN_memory1 --dataset officehome --source RealWorld --target Product  --batch-size 32 --max_epoch 30 | tee Officehome_Memory1_banchmark_0729/officehome_DANN_R2P.txt