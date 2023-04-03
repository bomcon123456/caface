
DATA_ROOT='/lustre/scratch/client/scratch/guardpro/trungdt21/matching/data/caface/train'
PRECOMPUTE_TRAIN_REC='vit47788_centers'
BACKBONE_MODEL='/lustre/scratch/client/vinai/users/trungdt21/tmp/insightface/experiments/47788/ckpt/Backbone_Epoch_93_checkpoint.pth'
CENTER_PATH='/lustre/scratch/client/guardpro/trungdt21/matching/data/training/WebFace4M/features/vit47788_centers/center_vit_l_dp005_mask_005_ckpt.pth'

WANDB_MODE=offline python main.py \
          --prefix vit47788_test \
          --data_root ${DATA_ROOT} \
          --use_precompute_trainrec ${PRECOMPUTE_TRAIN_REC} \
          --start_from_model_statedict ${BACKBONE_MODEL} \
          --center_path ${CENTER_PATH} \
          --train_data_path WebFace4MRec \
          --gpus 8 \
          --wandb_tags vit_l_dp005_mask_005 \
          --arch vit_l_dp005_mask_005 \
          --tpus 0 \
          --num_workers 16 \
          --batch_size 512 \
          --val_batch_size 64 \
          --num_images_per_identity 32 \
          --freeze_model \
          --aggregator_name style_norm_srm \
          --intermediate_type style \
          --style_index 3,5 \
          --decoder_name catv9_g4_conf512_small \
          --center_loss_lambda 1.0 \
          --limit_train_batches 1.0 \
          --same_aug_within_group_prob 0.75 \
          --datafeed_scheme dual_multi_v1 \
          --epochs 10 \
          --lr 1e-3 \
          --optimizer_type adamw \
          --lr_milestones 6,9 \
          --lr_scheduler step \
          --weight_decay 5e-4 \
          --img_aug_scheme v3 \
          --use_memory \
          --memory_loss_lambda 1.0
