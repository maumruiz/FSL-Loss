python train.py --gpu $1 --dataset Cars --way 5 --shot 1 --train_way 30 --modules ICN_Loss --losses cross,fullicnn --tag 7
python train.py --gpu $1 --dataset Cars --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses cross,fullicnn --tag 7

python train.py --gpu $1 --dataset Cars --way 5 --shot 1 --train_way 30 --modules ICN_Loss --losses fullicnn --tag 8
python train.py --gpu $1 --dataset Cars --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses fullicnn --tag 8