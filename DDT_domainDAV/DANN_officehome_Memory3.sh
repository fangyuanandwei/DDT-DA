mkdir Memory3_banchmark
mkdir Memory3_banchmark/officehome_0320_DANN

CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source Art --target Clipart --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_Art2Clipart.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source Art --target Product --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_Art2Clipart.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source Art --target RealWord --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_Art2Clipart.txt

CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source Clipart --target Art --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_Clipart2Art.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source Clipart --target Product --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_Clipart2Product.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source Clipart --target RealWord --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_Clipart2RealWord.txt

CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source Product --target Art --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_Product2Art.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source Product --target Clipart --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_Product2Clipart.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source Product --target RealWord --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_Product2RealWord.txt

CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source RealWord --target Art --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_RealWord2Art.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source RealWord --target Clipart --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_RealWord2Clipart.txt
CUDA_VISIBLE_DEVICES=7 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory3.py --day 0320 --time 0001 --method DANN_memory3 --dataset officehome --source RealWord --target Product --batch-size 32 --max_epoch 20 | tee Memory3_banchmark/officehome_0320_DANN/officehome_RealWord2Product.txt
