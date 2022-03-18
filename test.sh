modelfile=$1
subset=$2
dataset=$3
python3 fairseq/scripts/average_checkpoints.py --inputs checkpoints/$modelfile --num-epoch-checkpoints 5 --output checkpoints/$modelfile/average-epoch.pt
fairseq-generate data-bin/$dataset --gen-subset $subset --path checkpoints/$modelfile/average-epoch.pt --beam 4 --remove-bpe --lenpen 0.6 --no-progress-bar --log-format json --left-pad-source False | tee result/pred_${modelfile}_${subset}.txt
