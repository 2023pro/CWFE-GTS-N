# coding: utf-8
import numpy as np
from src.train_and_evaluate import *
from src.models import *
import torch
import time
import torch.optim
from src.expressions_transfer import *
from tqdm import tqdm
import numpy

torch.cuda.set_device(2)

batch_size = 64
embedding_size = 128
hidden_size = 768
n_epochs = 80
learning_rate = 1e-3
bert_learning_rate = 5e-5

beam_size = 5
n_layers = 2
n_layer = 6
dropout = 0.5
if_N_label_loss = True #是否使用负标签损失

seed = 711
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

transformer_name = "hfl/chinese-electra-180g-base-discriminator"

data = load_ape_data("data/train.ape.json")
pairs, generate_nums, copy_nums ,all_seq_words_nature,all_BO,ID = transfer_ape_num(data, id_file="data/Ape_clean2_train.txt") #Ape_clean2就是论文中的Ape_clean~
temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_trained = temp_pairs

data = load_ape_data("data/test.ape.json")
pairs, _, _ ,all_test_seq_words_nature,all_BO_test,ID_test= transfer_ape_num(data, id_file="data/Ape_clean2_test.txt")
temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_tested = temp_pairs

input_lang, output_lang, words_nature_lang,train_pairs, test_pairs = prepare_data_bert(pairs_trained, pairs_tested, 5, generate_nums,
                                                                     copy_nums, 'weight/electra/vocab.txt',all_seq_words_nature,all_test_seq_words_nature,
                                                                                       all_BO,all_BO_test, ID,ID_test,tree=True)

print(output_lang.index2word)
print(output_lang.word2index)

O_L = output_lang.word2index

encoder = Encoder_bert_rnn(input_dim=input_lang.n_words,w_n_dim = words_nature_lang.n_words, emb_dim=hidden_size, hid_dim=hidden_size, n_layers=n_layer,
              model_name=transformer_name, dropout=dropout)

predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))

generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)

merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

#encoder_parameter = torch.load("encoder_models/encoder", map_location=torch.device('cpu'))
'''encoder_parameter = torch.load("encoder_models/encoder")
encoder.load_state_dict(encoder_parameter)'''
encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=bert_learning_rate)
predict_optimizer = torch.optim.AdamW(predict.parameters(), lr=learning_rate)
generate_optimizer = torch.optim.AdamW(generate.parameters(), lr=learning_rate)
merge_optimizer = torch.optim.AdamW(merge.parameters(), lr=learning_rate)

encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.5, patience=4, verbose=True, threshold=0.1, threshold_mode='rel')
predict_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(predict_optimizer, mode='min', factor=0.5, patience=4, verbose=True, threshold=0.1, threshold_mode='rel')
generate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(generate_optimizer, mode='min', factor=0.5, patience=4, verbose=True, threshold=0.1, threshold_mode='rel')
merge_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(merge_optimizer, mode='min', factor=0.5, patience=4, verbose=True, threshold=0.1, threshold_mode='rel')

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

generate_num_ids = []
best_val_acc = 0

for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, \
    num_size_batches,words_nature_input_batches,words_nature_input_length,BO_input_batches,BO_length= prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()

    for idx in tqdm(range(len(input_lengths))):
        loss= train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang,
            num_pos_batches[idx], words_nature_input_batches[idx], words_nature_input_length[idx],
            BO_input_batches[idx], BO_length[idx],O_L,if_N_label_loss)
        loss_total += loss

    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    if epoch == 0 or epoch == 10 or epoch >= 20:
        start = time.time()
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        for test_batch in test_pairs:
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                     merge, output_lang, test_batch[5], test_batch[7], test_batch[9], test_batch[10], beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang,
                                                              test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        if (float(value_ac) / eval_total) > best_val_acc:
            best_val_acc = float(value_ac) / eval_total
            torch.save(encoder.state_dict(), "/home1/sq/CWFE-GTS-N_Ape/models/encoder")
            torch.save(predict.state_dict(), "/home1/sq/CWFE-GTS-N_Ape/models/predict")
            torch.save(generate.state_dict(), "/home1/sq/CWFE-GTS-N_Ape/models/generate")
            torch.save(merge.state_dict(), "/home1/sq/CWFE-GTS-N_Ape/models/merge")
            epoch_best = epoch
        print("best_val_acc:", best_val_acc)

    encoder_scheduler.step(loss_total / len(input_lengths))
    predict_scheduler.step(loss_total / len(input_lengths))
    generate_scheduler.step(loss_total / len(input_lengths))
    merge_scheduler.step(loss_total / len(input_lengths))



