#datapath=/home/robert.lim/datasets/mvtec
datapath='/home/robert.lim/main/config.json' # config.json 과 dataset name 을 이용해서 data_path 를 자동 load 함
#datasets=(Crop1)
datasets=(SeoulOKNG)
#datasets=(PCBNG PCBNG_0.01 PCBNG_0.1 Balanced-PCBNG Imbalanced_OK-PCBNG Imbalanced_NG-PCBNG BODY_ESOC_EXCEPT4_7_0.01 BODY_ESOC4_0.01 BODY_ESOC7_0.01 SIDE_ESOC4_0.01 LEAD_ALL_0.01 TERRACE_ESOC11_0.01 TERRACE_ESOC4_0.01 TERRACE_ESOC7_0.01 TRAIN_corner_0.01 SeoulOKNG)
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

# NOTE 원래 patchsize 는 3 임
# NOTE 원래 batch_size 는 8 임
#--meta_epochs 40 \
#--gan_epochs 4 \
function run_simplenet {
CUDA_VISIBLE_DEVICES=3 python3 main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_mvtec \
--log_project MVTecAD_Results \
--results_path lg_results \
--run_name run \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 6 \
--meta_epochs 40 \
--gan_epochs 4 \
--embedding_size 256 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
--onnx $onnx \
--mainmodel vig \
dataset \
--batch_size 1 \
--resize 329 \
--imagesize 288 "${dataset_flags[@]}" lg $datapath
}

onnx="no"
run_simplenet