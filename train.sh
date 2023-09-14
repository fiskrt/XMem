num_gpus=2 # ~5 gb vram/batch => batch 8 need 40 gb (ex two 20+ gb cards)
batch_size=8
name=cleanstart_all_onlybbox_searchsize5_timetest

echo starting $name.txt
gpu_id=4,5
num_workers=4
export CUDA_VISIBLE_DEVICES=$gpu_id

#echo "Using processors:"$((128-num_workers*gpu_id-num_workers+1))-$((128-num_workers*gpu_id)) #  $((num_workers*gpu_id))-$((num_workers*gpu_id+num_workers-1))
#taskset -c $((128-num_workers*gpu_id-num_workers+1))-$((128-num_workers*gpu_id)) \
taskset -c 80-87 \
 	torchrun --nproc_per_node=$num_gpus \
	--master_port $(shuf -i 22000-26000 -n 1) \
	train.py \
	--exp_id $name \
	--stage 3 \
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
	--dice_numerator_smoothing \
	--save_network_interval 50000 \
	--save_checkpoint_interval 50000 \
	--temporal_search_size 5 \
	--first_frame_bbox \
	&> log/$name.txt


#	--no_temporal_loss \
#	--no_pairwise_loss \
#	--load_network "saves_models/XMem-s0.pth" \
#	--train_with_first_frame \
#	--load_checkpoint "saves_models/Jul31_17.52.02_all_nosmooth_trainfirstframe_s3/Jul31_17.52.02_all_nosmooth_trainfirstframe_s3_checkpoint_92095.pth" \
#	--train_on_mose \
#	--original_loss \
#	--debug
#	--use_ratio_loss \
#	--temporal_loss_scale 0.05 \
#	--pairwise_loss_scale 0.5 \
#	--pairwise_warmup_steps 5000 \
#	--num_loss_frames 2 \
#	--log_image_interval 25 \
#	--no_projection_loss \
#	--log_text_interval 10 \
