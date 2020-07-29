#!/bin/bash

python interact_mmi.py \
	--device=0 \
	--dialogue_model_path=./my_model/dialogue \
	--mmi_model_path=./my_model/mmi \
	--vocab_path=./vocab/vocab_chat.txt \
	--repetition_penalty=2 \
