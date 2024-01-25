mkdir MemoryResult_VisDA_MultiGPU
#mkdir MemoryResult_VisDA_MultiGPU/Visda_0321_BCDM
#CUDA_VISIBLE_DEVICES=4,5 /raid/huangl05/anaconda3.6/bin/python3.6 MemoryGuid_MemoryMain_MultiGPU.py --day 0321 --time 1200 --method BCDM --dataset visda --source synthetic --target real  --lr 0.003 --max_epoch 20 --initepoch True | tee MemoryResult_VisDA_MultiGPU/Visda_0321_BCDM/visda_s2r.txt
mkdir MemoryResult_VisDA_MultiGPU/Visda_0321_MCD
CUDA_VISIBLE_DEVICES=4,5 /raid/huangl05/anaconda3.6/bin/python3.6 MemoryGuid_MemoryMain_MultiGPU.py --day 0321 --time 1200 --method MCD --dataset visda --source synthetic --target real   --lr 0.003 --max_epoch 20 --initepoch True | tee MemoryResult_VisDA_MultiGPU/Visda_0321_MCD/visda_s2r.txt
