num_gpus=1 # ~5 gb vram/batch => batch 8 need 40 gb (ex two 20+ gb cards)
batch_size=8

torchrun --nproc_per_node=$num_gpus \
		--master_port $(shuf -i 22000-26000 -n 1) \
	train.py \
	--exp_id NAME_OF_EXPERIMENT \
	--stage 3 \
	--s3_batch_size $batch_size \
	--s3_lr 1e-5 \
	--s3_iterations 100000 \
	--num_workers 16 \
	--pairwise_color_threshold 0.3 \
	--temporal_color_threshold 0.005 \
	--temporal_theta 0.3 \
	--projection_loss_scale 10.0 \
	--pairwise_loss_scale 0.05 \
	--temporal_loss_scale 0.5 \
	--dice_numerator_smoothing \
	--detach_temporal_loss \
#	--load_checkpoint "saves_models/May16_18.12.09_all_03pl03_temp001_05theta03_alpha10_700902_s3/May16_18.12.09_all_03pl03_temp001_05theta03_alpha10_700902_s3_checkpoint_110000.pth" 
#	--no_pairwise_loss \
#	--no_temporal_loss \
#	--use_ratio_loss \
#	--first_frame_bbox \
#	--temporal_loss_scale 0.05 \
#	--pairwise_loss_scale 0.5 \
#	--pairwise_warmup_steps 5000 \
#	--num_loss_frames 2 \
#	--log_image_interval 25 \
#	--no_projection_loss \
#	--log_text_interval 10 \