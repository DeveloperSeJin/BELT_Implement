#!/bin/bash



# ['DELTA', 'wo_diffusion', 'DConv', 'EEGNet', 'wo_pretrained','main_diffusion', 'full_diffusion']
# [pretrain, DConv_pretrain, EEGNet_pretrain, wo_diffusion_pretrain]
# subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']

batch=32
noise="sqrt"
time_step=1000
subject=cross
# subjects=( 'ZAB' 'ZDM' 'ZGW' 'ZJM' 'ZJN' 'ZJS' 'ZKB' 'ZKH' 'ZKW' 'ZMG' 'ZPH' )
# subjects=('YDR' 'YFR' 'YFS' 'YAC' 'YDG' 'YHS' 'YMD' 'YLS' 'YRH' 'YRK' 'YSD' 'YRP' 'YSL' 'YTL' 'YAG' 'YAK' 'YIS' 'YMS' )



python3 train_decoding.py \
  --model_name BrainTranslator \
  --task_name taskNRv2 \
  --one_step \
  --pretrained \
  --not_load_step1_checkpoint \
  --num_epoch_step1 20 \
  --num_epoch_step2 30 \
  -lr1 0.00005 \
  -lr2 0.0000005 \
  -b 32 \
  -s ./checkpoints/decoding \
  -setting "$subject" \
  -cuda cuda:2

# 활성화 옵션 메모:
# -con \
# -geo\
# -kl \