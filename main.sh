python main.py --device cuda:1 \
               --model microsoft/deberta-base \
               --batch-size 8 \
               --epochs 2 \
               --scheduler expon \
               --lr 2e-5 \
               --n-fold 5 \
               --gamma 0.9995 \
               --gradient-accumulation-steps 1 \
               --train True \
               --test True

python main.py --device cuda:1 \
               --model microsoft/deberta-base \
               --batch-size 8 \
               --epochs 2 \
               --scheduler cosin \
               --lr 2e-5 \
               --n-fold 5 \
               --gamma 0.9995 \
               --gradient-accumulation-steps 1 \
               --train True \
               --test True

python main.py --device cuda:1 \
               --model microsoft/deberta-base \
               --batch-size 8 \
               --epochs 2 \
               --scheduler cycle \
               --lr 2e-5 \
               --n-fold 5 \
               --gamma 0.9995 \
               --gradient-accumulation-steps 1 \
               --train True \
               --test True

python main.py --device cuda:1 \
               --model microsoft/deberta-base \
               --batch-size 8 \
               --epochs 2 \
               --scheduler lambda \
               --lr 2e-5 \
               --n-fold 5 \
               --gamma 0.9995 \
               --gradient-accumulation-steps 1 \
               --train True \
               --test True

python main.py --device cuda:1 \
               --model microsoft/deberta-base \
               --batch-size 8 \
               --epochs 10 \
               --scheduler cosin \
               --lr 2e-5 \
               --n-fold 5 \
               --gamma 0.9995 \
               --gradient-accumulation-steps 1 \
               --train True \
               --test True