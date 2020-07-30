# -*- coding: utf-8 -*-
import argparse
import thulac
import json

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', default='../data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--vocab_file', default='vocab_processed.txt', type=str, required=False, help='生成vocab链接')
    parser.add_argument('--vocab_size', default=50000, type=int, required=False, help='词表大小')
    parser.add_argument('--char_level', action='store_true')
    args = parser.parse_args()

    lac = thulac.thulac(seg_only=True)
    tokenizer = Tokenizer(num_words=args.vocab_size, char_level=args.char_level)
    print('args:\n' + args.__repr__())
    print('This script is extremely slow especially for large corpus. Take a break.')
    
    f = open(args.raw_data_path, mode='r', encoding='utf-8')
    s = f.read()
    if '\r\n' in s:
        lines = s.split('\r\n')
    else:
        lines = s.split('\n')
    
    print('len of lines =', len(lines))
    
    for i, line in enumerate(tqdm(lines)):
        lines[i] = lac.cut(line, text=True)
    

    tokenizer.fit_on_texts(lines)
    vocab = list(tokenizer.index_word.values())
    pre = ['[SEP]', '[CLS]', '[MASK]', '[PAD]', '[UNK]']
    vocab = pre + vocab
    with open(args.vocab_file, mode='w+', encoding='utf-8') as f:
        for word in vocab[:args.vocab_size + 5]:
            f.write(word + '\n')

if __name__ == "__main__":
    main()
