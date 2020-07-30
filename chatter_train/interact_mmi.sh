python interact_mmi.py \
    --dialogue_model_path=./dialogue_model/model_epoch3/ \
    --mmi_model_path=./mmi_model/model_epoch3 \
    --device=2,3 \
    --voca_path=./data/vocab_chat.txt \
    --repetition_penalty=1.5 \
    --batch_size=5 \
    --no_cuda
