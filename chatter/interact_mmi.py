import os
from os.path import join, exists
import argparse
import copy
from datetime import datetime
import logging

import torch
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F


def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--dialogue_model_path', default='dialogue_model/', type=str, required=False,
                        help='dialogue_model路径')
    parser.add_argument('--mmi_model_path', default='mmi_model/', type=str, required=False,
                        help='互信息mmi_model路径')
    parser.add_argument('--chatlog_path', default="chatlog/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--candidate_num', type=int, default=5, help='批量生成response，然后经过MMI模型进行筛选')
    parser.add_argument('--debug', action='store_true', help='指定该参数，可以查看生成的所有候选的reponse，及其loss')
    return parser.parse_args()


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def main():
    args = set_interact_args()
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    if args.chatlog_path:
        if not os.path.exists(args.chatlog_path):
            os.makedirs(args.chatlog_path)
        samples_file = open(args.chatlog_path + '/chatlog.txt', 'a', encoding='utf-8')
        samples_file.write("聊天紀錄{}:\n".format(datetime.now()))
        # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print(u'開始和chatbot聊天，輸入CTRL + Z以退出')

    while True:
        try:
            text = input("user:")
            if args.chatlog_path:
                samples_file.write("user:{}\n".format(text))
            history.append(tokenizer.encode(text))
            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            # 用于批量生成response，维度为(candidate_num,token_len)
            input_ids = [copy.deepcopy(input_ids) for _ in range(args.candidate_num)]

            curr_input_tensors = torch.tensor(input_ids).long().to(device)
            generated = []  # 二维数组，维度为(生成的response的最大长度，candidate_num)，generated[i,j]表示第j个response的第i个token的id
            finish_set = set()  # 标记是否所有response均已生成结束，若第i个response生成结束，即生成了sep_token_id，则将i放入finish_set
            # 最多生成max_len个token
            for _ in range(args.max_len):
                outputs = dialogue_model(input_ids=curr_input_tensors)
                next_token_logits = outputs[0][:, -1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for index in range(args.candidate_num):
                    for token_id in set([token_ids[index] for token_ids in generated]):
                        next_token_logits[index][token_id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                for next_token_logit in next_token_logits:
                    next_token_logit[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                # 判断是否有response生成了[SEP],将已生成了[SEP]的resposne进行标记
                for index, token_id in enumerate(next_token[:, 0]):
                    if token_id == tokenizer.sep_token_id:
                        finish_set.add(index)
                # 检验是否所有的response均已生成[SEP]
                finish_flag = True  # 是否所有的response均已生成[SEP]的token
                for index in range(args.candidate_num):
                    if index not in finish_set:  # response批量生成未完成
                        finish_flag = False
                        break
                if finish_flag:
                    break
                generated.append([token.item() for token in next_token[:, 0]])
                # 将新生成的token与原来的token进行拼接
                curr_input_tensors = torch.cat((curr_input_tensors, next_token), dim=-1)
            candidate_responses = []  # 生成的所有候选response
            for batch_index in range(args.candidate_num):
                response = []
                for token_index in range(len(generated)):
                    if generated[token_index][batch_index] != tokenizer.sep_token_id:
                        response.append(generated[token_index][batch_index])
                    else:
                        break
                candidate_responses.append(response)

            # mmi模型的输入
            if args.debug:
                print("candidate response:")
                samples_file.write("candidate response:\n")

            min_loss = float('Inf')
            best_response = ""
            for response in candidate_responses:
                mmi_input_id = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
                mmi_input_id.extend(response)
                mmi_input_id.append(tokenizer.sep_token_id)

                for history_utr in reversed(history[-args.max_history_len:]):
                    mmi_input_id.extend(history_utr)
                    mmi_input_id.append(tokenizer.sep_token_id)
                mmi_input_tensor = torch.tensor(mmi_input_id).long().to(device)
                out = mmi_model(input_ids=mmi_input_tensor, labels=mmi_input_tensor)
                loss = out[0].item()

                if args.debug:
                    text = tokenizer.convert_ids_to_tokens(response)
                    print("{} loss:{}".format("".join(text), loss))
                    samples_file.write("{} loss:{}\n".format("".join(text), loss))

                if loss < min_loss:
                    best_response = response
                    min_loss = loss

            history.append(best_response)
            text = tokenizer.convert_ids_to_tokens(best_response)
            print("chatbot:" + "".join(text))
            if args.chatlog_path:
                samples_file.write("chatbot:{}\n".format("".join(text)))

        except KeyboardInterrupt:
            if args.chatlog_path:
                samples_file.write("\n")
                samples_file.close()
            break



if __name__ == '__main__':
    main()
