#CLEF
#DANN_memory2
CUDA_VISIBLE_DEVICES=5 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory2.py --day 0724 --time 1619 --method DANN_memory2 --dataset CLEF --source i_split --target p_split  --batch-size 32 --max_epoch 20
CUDA_VISIBLE_DEVICES=5 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory2.py --day 0724 --time 1619 --method DANN_memory2 --dataset CLEF --source i_split --target c_split  --batch-size 32 --max_epoch 20

CUDA_VISIBLE_DEVICES=5 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory2.py --day 0724 --time 1619 --method DANN_memory2 --dataset CLEF --source p_split --target i_split  --batch-size 32 --max_epoch 20
CUDA_VISIBLE_DEVICES=5 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory2.py --day 0724 --time 1619 --method DANN_memory2 --dataset CLEF --source p_split --target c_split  --batch-size 32 --max_epoch 20

CUDA_VISIBLE_DEVICES=5 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory2.py --day 0724 --time 1619 --method DANN_memory2 --dataset CLEF --source c_split --target i_split  --batch-size 32 --max_epoch 20
CUDA_VISIBLE_DEVICES=5 /raid/huangl05/anaconda3.6/bin/python3.6 DANN_memory2.py --day 0724 --time 1619 --method DANN_memory2 --dataset CLEF --source c_split --target p_split  --batch-size 32 --max_epoch 20