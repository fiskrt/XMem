num_gpus=2 # ~5 gb vram/batch => batch 8 need 40 gb (ex two 20+ gb cards)
batch_size=8
name=s0_onlyproj

gpu_id=6,7
num_workers=4
export CUDA_VISIBLE_DEVICES=$gpu_id

#echo "Using processors:"$((128-num_workers*gpu_id-num_workers+1))-$((128-num_workers*gpu_id)) #  $((num_workers*gpu_id))-$((num_workers*gpu_id+num_workers-1))
#taskset -c $((128-num_workers*gpu_id-num_workers+1))-$((128-num_workers*gpu_id)) \
taskset -c 64-71 \
 	torchrun --nproc_per_node=$num_gpus \
	--master_port $(shuf -i 22000-26000 -n 1) \
	train.py \
	--exp_id $name \
	--train_with_first_frame \
	--stage 0 \
	--s3_batch_size $batch_size \
	--s3_lr 1e-5 \
	--s3_iterations 100000 \
	--num_workers $num_workers \
	--pairwise_color_threshold 0.3 \
	--temporal_color_threshold 0.01 \
	--temporal_theta 0.3 \
	--projection_loss_scale 10.0 \
	--pairwise_loss_scale 0.05 \
	--temporal_loss_scale 0.5 \
	--detach_temporal_loss \
	--save_network_interval 50000 \
	--save_checkpoint_interval 100000 \
	--dice_numerator_smoothing \
	--no_pairwise_loss \
	--no_temporal_loss \
	&> log/$name.txt
#	--load_network "saves_models/XMem-s01.pth" \
#	--train_on_mose \
#	--load_checkpoint "saves_models/Jul26_01.12.51_all_train_with_first_frame_s3/Jul26_01.12.51_all_train_with_first_frame_s3_checkpoint_100000.pth" \
#	--original_loss \
#	--first_frame_bbox \
#	--debug
#	--use_ratio_loss \
#	--temporal_loss_scale 0.05 \
#	--pairwise_loss_scale 0.5 \
#	--pairwise_warmup_steps 5000 \
#	--num_loss_frames 2 \
#	--log_image_interval 25 \
#	--no_projection_loss \
#	--log_text_interval 10 \
