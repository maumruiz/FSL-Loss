python train.py --gpu $1 --dataset Dogs --way 5 --shot 1 --train_way 30 --modules ICN_Loss --losses suppicnn --tag 3
python train.py --gpu $1 --dataset Dogs --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses suppicnn --tag 3

python train.py --gpu $1 --dataset Dogs --way 5 --shot 1 --train_way 30 --modules ICN_Loss --losses suppicnn,queryicnn --tag 4
python train.py --gpu $1 --dataset Dogs --way 5 --shot 5 --train_way 20 --modules ICN_Loss --losses suppicnn,queryicnn --tag 4