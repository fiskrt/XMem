import os



# We need the path to the trained model

# First eval.py the model using either D17 or ytVOS

#python eval.py --model saves/model_to_be_evaluated.pth 
# --output ./results/d17_proj_scratch_NOCROP_s3_25k 
# --dataset D17

os.system('python eval.py'
' --model saves/Mar19_00.09.31_temporal_with_mean_lambda0125_s3/Mar19_00.09.31_temporal_with_mean_lambda0125_s3_175000.pth'
' --output ./results/y19_temporal_s3_175k --dataset Y19')
print('doing 175k evaluation!')

# If davis, use
#python evaluation_method.py --task semi-supervised 
# --results_path ../XMem/results/d17_proj_scratch_NOCROP_s3_50k/ 
# --davis_path ../DAVIS/2017/trainval/


# if youtubeVOS 19 use: https://codalab.lisn.upsaclay.fr/competitions/7683#participate-submit_results