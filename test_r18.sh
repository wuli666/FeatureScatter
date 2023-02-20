export PYTHONPATH=./:$PYTHONPATH
model_dir=./checkpiont/r18_cifar10/
CUDA_VISIBLE_DEVICES=0 python3 test_r18.py \
    --model_dir=$model_dir \
    --init_model_pass=0 \
    --attack=True \
    --attack_method_list=natural-fgsm-pgd-cw \
    --dataset=cifar10 \
    --batch_size_test=80 \
    --resume
