export PYTHONPATH=./:$PYTHONPATH
model_dir=./checkpiont/wrn28_cifar10/
CUDA_VISIBLE_DEVICES=0 python3 test_wrn.py \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --attack=True \
    --attack_method_list=natural-fgsm-pgd-cw \
    --dataset=cifar10 \
    --batch_size_test=80 \
    --resume
