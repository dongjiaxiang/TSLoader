export CUDA_VISIBLE_DEVICES=0

python -u ./PatchTST_self_supervised/patchtst_pretrain.py --dset weather --mask_ratio 0.4
