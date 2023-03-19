import os

#os.system('python -m torch.distributed.run --master_port 25763 --nproc_per_node=1 train.py --exp_id dav_yt_only_dice_half_lr_night --stage 3 --s3_lr 5e-06')
#os.system('python -m torch.distributed.run --master_port 25763 --nproc_per_node=1 train.py --exp_id davis_yt_only_proj --stage 3 --load_network saves/XMem-s0.pth')


# pretrain
#os.system('python -m torch.distributed.run --master_port 25763 --nproc_per_node=1 train.py --exp_id s0_night --stage 0')


# train pre-trained
#os.system('python -m torch.distributed.run --master_port 25763 --nproc_per_node=1'
#     ' train.py --exp_id s03_only_dice_night2_lr07 --stage 3 --s3_lr 1e-07 --load_network saves/Feb07_01.28.07_s0_night_s0/Feb07_01.28.07_s0_night_s0_25000.pth')

# pairwise/projection loss from scratch
#os.system('python -m torch.distributed.run --master_port 25763 --nproc_per_node=1'
#     ' train.py --exp_id pair_proj_fixdice --stage 3 --s3_lr 1e-06 --s3_batch_size 3')


os.system('python -m torch.distributed.run --master_port 25763 --nproc_per_node=1'
     ' train.py --exp_id temporal_with_mean_lambda0125 --stage 3 --log_image_interval 1000' # --s3_lr 1e-06')
     ' --s3_batch_size 2 --s3_lr 1e-05'
     ' --s3_iterations 200000'
     ' --load_checkpoint saves/Mar17_16.23.42_temporal_with_mean_lambda0125_s3/Mar17_16.23.42_temporal_with_mean_lambda0125_s3_checkpoint_100000.pth'
#     ' --load_network saves/Feb22_00.04.58_pair_scratch_NOCROP_s3/Feb22_00.04.58_pair_scratch_NOCROP_s3_25000.pth'
    # ' --load_network saves/Feb22_00.04.58_pair_scratch_NOCROP_s3/Feb22_00.04.58_pair_scratch_NOCROP_s3_25000.pth'
#     ' --load_network saves/XMem-s0.pth '
)