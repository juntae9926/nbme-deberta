python main.py --device cuda:1 \
               --model microsoft/deberta-base \
               --batch-size 16 \
               --epochs 5 \
               --lr 2e-5 \
               --n-fold 5 \
               --gradient-accumulation-steps 1 \
               --train True \
               --test True