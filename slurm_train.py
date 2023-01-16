import os

os.system('python -m torch.distributed.run --master_port 25763 --nproc_per_node=1 train.py '
            '--exp_id overnight_bl_dice_and_bce --stage 3 --load_network saves/XMem-s0.pth')