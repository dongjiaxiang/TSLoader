export CUDA_VISIBLE_DEVICES=0

python -u ./PatchTST_self_supervised/patchtst_pretrain.py --dset ts_mergedatasets --mask_ratio 0.4 --batch_size 512
