#office31
#BCDM
CUDA_VISIBLE_DEVICES=0 /raid/huangl05/anaconda3.6/bin/python3.6 MemoryGuid_MemoryMain_belta.py --day 3022 --time 1200 --method BCDM --dataset visda --source synthetic --target real  --lr 0.003 --max_epoch 20 --initepoch True
#MCD
CUDA_VISIBLE_DEVICES=0 /raid/huangl05/anaconda3.6/bin/python3.6 MemoryGuid_MemoryMain_belta.py --day 3022 --time 1200 --method MCD --dataset visda --source synthetic --target real   --lr 0.003 --max_epoch 20 --initepoch True
