# encoding:utf-8
# author: xinhchen (xinhchen2-c@my.cityu.edu.hk)
# reference: zxding (d.z.x@qq.com)

import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb, time, logging, datetime


def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))


def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r', encoding='utf-8')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend([emotion] + clause.split())

    # vocabulary
    words = set(words)
    # Dictionary of {word: word_id}
    word_idx = dict((c, k + 1) for k, c in enumerate(words))
    # Dictionary of {word_id: word}
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))

    w2v = {}
    inputFile2 = open(embedding_path, 'r', encoding='utf-8')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    # positional embedding (Not used in our paper)
    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)

    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos


def load_data(input_file, word_idx, label2id, emo2id, max_doc_len = 75, max_sen_len = 45):
    print('load data_file: {}'.format(input_file))
    y_pairs, x, sen_len, doc_len, y, l, el = [], [], [], [], [], [], []
    doc_id = []

    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        pos, cause = zip(*pairs)

        y_tmp = np.zeros((max_doc_len, 19), np.int32)
        l_tmp = np.zeros((max_doc_len, 4), np.int32)
        el_tmp = np.zeros((max_doc_len, 7))
        sen_len_tmp = np.zeros(max_doc_len, dtype=np.int32)
        x_tmp = np.zeros((max_doc_len, max_sen_len), dtype=np.int32)
        emolabeldict = {}

        for i in range(d_len):
            [sen_id, emolabel, _, words] = inputFile.readline().strip().split(',')
            emolabeldict[i] = emolabel
            sen_id = int(sen_id)
            if sen_id not in pos and sen_id not in cause:
                y_tmp[sen_id-1][label2id["O"]] = 1
                l_tmp[sen_id-1][0] = 1 # 0 is for "O"
                el_tmp[sen_id-1][emo2id["O"]] = 1
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])

        # Create emotion labels and cause labels separately for each clause
        for pair in pairs:
            if pair[0] == pair[1]:
                y_tmp[pair[0]-1][label2id["B-"+emolabeldict[pair[0]-1]]] = 1
                l_tmp[pair[0]-1][3] = 1 # 3 stands for both e and c
            else:
                if l_tmp[pair[0]-1][3] != 1:
                    y_tmp[pair[0]-1][label2id["E-"+emolabeldict[pair[0]-1]]] = 1
                    l_tmp[pair[0]-1][1] = 1 # 1 stands for e
                y_tmp[pair[1]-1][label2id["C-"+emolabeldict[pair[0]-1]]] = 1
                l_tmp[pair[1]-1][2] = 1 # 2 stands for c
            el_tmp[pair[0]-1][emo2id[emolabeldict[pair[0]-1]]] = 1
            el_tmp[pair[1]-1][emo2id[emolabeldict[pair[0]-1]]] = 1

        y.append(y_tmp)
        l.append(l_tmp)
        el.append(el_tmp)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)

    y, l, el, x, sen_len, doc_len = map(np.array, [y, l, el, x, sen_len, doc_len])
    for var in ['y', 'l', 'el', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return doc_id, y_pairs, x, sen_len, doc_len, y, l, el


def acc_prf(pred_y, true_y, doc_len, average='binary'):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1


def mapcausepos(pred_y, true_y):
    pred_y = np.argmax(pred_y, 2) if len(pred_y.shape) > 2 else pred_y
    true_y = np.argmax(true_y, 2) if len(true_y.shape) > 2 else true_y
    pred_y_cause = np.zeros((pred_y.shape[0], pred_y.shape[1], 2), np.int32)
    true_y_cause = np.zeros((pred_y.shape[0], pred_y.shape[1], 2), np.int32)
    pred_y_pos = np.zeros((pred_y.shape[0], pred_y.shape[1], 2), np.int32)
    true_y_pos = np.zeros((pred_y.shape[0], pred_y.shape[1], 2), np.int32)

    pair_tp = 0
    pair_fp = 0
    pair_fn = 0
    for i in range(pred_y.shape[0]): # should be batch_size
        pred_cause = []
        pred_emo = []
        true_cause = []
        true_emo = []
        pred_pair = []
        true_pair = []
        for j in range(pred_y.shape[1]): # should be max_doc_len
            if pred_y[i][j] >= 7 and pred_y[i][j] <= 12:
                pred_cause.append((j, pred_y[i][j]-6))
            elif pred_y[i][j] >= 1 and pred_y[i][j] < 7:
                pred_emo.append((j, pred_y[i][j]))
            elif pred_y[i][j] >= 13:
                pred_cause.append((j, pred_y[i][j]-12))
                pred_emo.append((j, pred_y[i][j]-12))

            if true_y[i][j] >= 7 and true_y[i][j] <= 12:
                true_cause.append((j, true_y[i][j]-6))
            elif true_y[i][j] >= 1 and true_y[i][j] < 7:
                true_emo.append((j, true_y[i][j]))
            elif true_y[i][j] >= 13:
                true_cause.append((j, true_y[i][j]-12))
                true_emo.append((j, true_y[i][j]-12))

            pred_y_cause[i][j][int(pred_y[i][j] >= 7)] = 1
            true_y_cause[i][j][int(true_y[i][j] >= 7)] = 1
            pred_y_pos[i][j][int(pred_y[i][j] >= 1 and pred_y[i][j] < 7 or pred_y[i][j] >= 13)] = 1
            true_y_pos[i][j][int(true_y[i][j] >= 1 and true_y[i][j] < 7 or true_y[i][j] >= 13)] = 1

        if len(pred_cause) != 0 and len(pred_emo) == 0 or len(pred_cause) == 0 and len(pred_emo) != 0:
            continue
        else:
            for cause in pred_cause:
                for emo in pred_emo:
                    if cause[1] == emo[1]:
                        pred_pair.append([cause[0], emo[0]])
            for cause in true_cause:
                for emo in true_emo:
                    if cause[1] == emo[1]:
                        true_pair.append([cause[0], emo[0]])
            tmp_tp = 0
            for pair in pred_pair:
                if pair in true_pair:
                    tmp_tp += 1

            pair_tp += tmp_tp
            pair_fp += len(pred_pair) - tmp_tp
            pair_fn += len(true_pair) - tmp_tp

    pred_y_cause = np.argmax(pred_y_cause, 2)
    true_y_cause = np.argmax(true_y_cause, 2)
    pred_y_pos = np.argmax(pred_y_pos, 2)
    true_y_pos = np.argmax(true_y_pos, 2)

    pair_pre = 0.
    pair_rec = 0.
    pair_f1 = 0.
    pair_pre = pair_tp / (pair_tp + pair_fp + 1e-6)
    pair_rec = pair_tp / (pair_tp + pair_fn + 1e-6)
    pair_f1 = 2 * pair_pre * pair_rec / (pair_pre + pair_rec + 1e-6)

    return pred_y_cause, true_y_cause, pred_y_pos, true_y_pos, pair_pre, pair_rec, pair_f1


def get_logger(log_dir, scope):
    log_file = log_dir + "/" + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-'))  + scope + ".log"
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    # log into file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d: %H %M %S'))

    return logger


def loggercloser(arg_logger):
    handlers = arg_logger.handlers[:]
    for handler in handlers:
        handler.close()
        arg_logger.removeHandler(handler)
