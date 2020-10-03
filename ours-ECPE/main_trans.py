# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com


import numpy as np
import tensorflow as tf
import transformer as trans
from sklearn.model_selection import KFold
import sys, os, time, codecs, pdb

from utils.tf_funcs import *
from utils.prepare_data import *

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', '..\\data_rand\\w2v_200.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 45, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per document')
tf.app.flags.DEFINE_integer('max_cau_num', 3, 'max number of causes per document')
## model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 19, 'number of distinct class')
tf.app.flags.DEFINE_integer('n_label', 4, 'number of distinct label')
tf.app.flags.DEFINE_integer('n_elabel', 7, 'number of distinct elabel')
tf.app.flags.DEFINE_integer('n_aux', 2, 'number of distinct aux labels')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_dir', '..\\log', 'directory path of log file')
tf.app.flags.DEFINE_integer('max_to_keep', 5, 'maximum number of checkpoints')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 100, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.8, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 1e-4, 'l2 regularization')
tf.app.flags.DEFINE_integer('num_heads', 5, 'number of heads of self attention')
tf.app.flags.DEFINE_string("filter_sizes", "3,5", "CNN kernel size")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters per filter size")
tf.app.flags.DEFINE_float('cause', 1.000, 'lambda1')
tf.app.flags.DEFINE_float('pos', 1.000, 'lambda2')
tf.app.flags.DEFINE_integer("softmax_or_crf", 1, "mode of inference")
tf.app.flags.DEFINE_integer("num_layers", 2, "mode of inference")

tf.logging.set_verbosity(tf.logging.ERROR)


def build_model(word_embedding, x, sen_len, doc_len, keep_prob1, keep_prob2, y, l, el, filter_sizes, RNN=biLSTM):
    x = tf.nn.embedding_lookup(word_embedding, x)
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    sen_len = tf.reshape(sen_len, [-1])

    def get_s(inputs, name):
        with tf.name_scope('word_encode'):
            inputs = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope+'word_layer' + name)
        with tf.name_scope('word_attention'):
            sh2 = 2 * FLAGS.n_hidden
            w1 = get_weight_varible('word_att_w1' + name, [sh2, sh2])
            b1 = get_weight_varible('word_att_b1' + name, [sh2])
            w2 = get_weight_varible('word_att_w2' + name, [sh2, 1])
            s = att_var(inputs, sen_len, w1, b1, w2)
        s = tf.reshape(s, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
        return s
    s = get_s(inputs, name='label_word_encode')

    # classify causal label for each clause
    s_l = RNN(s, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'label_sentence_layer')
    with tf.variable_scope('label_prediction'):
        pred_label, reg_label = softmax_part(s_l, FLAGS.max_doc_len, 2 * FLAGS.n_hidden, keep_prob2, FLAGS.n_label, 'softmax_w_label', 'softmax_b_label')
        valid_l_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
        loss_label = -tf.reduce_sum(l * tf.log(pred_label)) / valid_l_num

    # classify emotion label for each clause
    s_el = RNN(s, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'elabel_sentence_layer')
    with tf.variable_scope('elabel_prediction'):
        pred_elabel, reg_elabel = softmax_part(s_el, FLAGS.max_doc_len, 2 * FLAGS.n_hidden, keep_prob2, FLAGS.n_elabel, 'softmax_w_elabel', 'softmax_b_elabel')
        valid_el_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
        loss_elabel = -tf.reduce_sum(el * tf.log(pred_elabel)) / valid_el_num

    s_c = tf.concat([s_l, s_el], 2)

    # Add a specific number of self-attention modules
    s_sen = trans_func(s_c, s_c, 4*FLAGS.n_hidden, 4*FLAGS.n_hidden, 'self-attention-0')
    for i in range(1, FLAGS.num_layers):
        s_sen = trans_func(s_sen, s_sen, 4*FLAGS.n_hidden, 4*FLAGS.n_hidden, 'self-attention-'+str(i))

    pred_class = tf.constant(0, tf.float32)
    reg_class = tf.constant(0, tf.float32)

    with tf.name_scope('context_prediction'):
        o_feature = 4 * FLAGS.n_hidden
        if FLAGS.softmax_or_crf:
            # SOFTMAX
            pred_class, reg_class = softmax_part(s_sen, FLAGS.max_doc_len, o_feature, keep_prob2, FLAGS.n_class, 'softmax_w_class', 'softmax_b_class')
            valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
            loss_class = -tf.reduce_sum(y * tf.log(pred_class)) / valid_num
        else:
            # CRF
            s1 = tf.reshape(s_sen, [-1, o_feature])
            s1 = tf.nn.dropout(s1, keep_prob=keep_prob2)
            w = get_weight_varible(name='CRF_w_class', shape=[o_feature, FLAGS.n_class])
            b = get_weight_varible(name='CRF_b_class', shape=[FLAGS.n_class])
            before_softmax = tf.matmul(s1, w) + b
            before_softmax = tf.reshape(before_softmax, [-1, FLAGS.max_doc_len, FLAGS.n_class])
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(before_softmax, tf.argmax(y, 2), doc_len)
            pred_class, viterbi_score = tf.contrib.crf.crf_decode(before_softmax, transition_params, doc_len)
            loss_class = tf.reduce_mean(-log_likelihood)

    reg = reg_class + reg_label + reg_elabel
    loss = loss_class + loss_label + loss_elabel

    return loss, pred_class, reg


def softmax_part(sent_e, n_num, n_feature, keep_prob, o_feature, w_name, b_name):
    s1 = tf.reshape(sent_e, [-1, n_feature])
    s1 = tf.nn.dropout(s1, keep_prob=keep_prob)
    w = get_weight_varible(name=w_name, shape=[n_feature, o_feature])
    b = get_weight_varible(name=b_name, shape=[o_feature])
    before_softmax = tf.matmul(s1, w) + b
    before_softmax = tf.reshape(before_softmax, [-1, n_num, o_feature])
    pred = tf.nn.softmax(before_softmax)
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg


# transformation function
def trans_func(senEncode_dis, senEncode, n_feature, out_units, scope_var):
    # Multi-head attention module
    senEncode_assist = trans.multihead_attention(queries=senEncode_dis,
                                            keys=senEncode_dis,
                                            values=senEncode,
                                            units_query=n_feature,
                                            num_heads=FLAGS.num_heads,
                                            dropout_rate=0,
                                            is_training=True,
                                            scope=scope_var)
    senEncode_assist = trans.feedforward_1(senEncode_assist, n_feature, out_units)
    return senEncode_assist


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, sen_len, doc_len, keep_prob1, keep_prob2, y, l, el, batch_size, test=False):
    for index in batch_index(len(y), batch_size, test):
        feed_list = [x[index], sen_len[index], doc_len[index], keep_prob1, keep_prob2, y[index], l[index], el[index]]
        yield feed_list, len(index)


def run():
    logger = get_logger(FLAGS.log_file_dir, FLAGS.scope)

    # Create label dictionary
    label2id = {}
    label2id["O"] = 0
    label2id["E-happiness"] = 1
    label2id["E-fear"] = 2
    label2id["E-surprise"] = 3
    label2id["E-sadness"] = 4
    label2id["E-disgust"] = 5
    label2id["E-anger"] = 6
    label2id["C-happiness"] = 7
    label2id["C-fear"] = 8
    label2id["C-surprise"] = 9
    label2id["C-sadness"] = 10
    label2id["C-disgust"] = 11
    label2id["C-anger"] = 12
    label2id["B-happiness"] = 13
    label2id["B-fear"] = 14
    label2id["B-surprise"] = 15
    label2id["B-sadness"] = 16
    label2id["B-disgust"] = 17
    label2id["B-anger"] = 18
    emo2id = {}
    emo2id["O"] = 0
    emo2id["happiness"] = 1
    emo2id["fear"] = 2
    emo2id["surprise"] = 3
    emo2id["sadness"] = 4
    emo2id["disgust"] = 5
    emo2id["anger"] = 6

    res_dir = '..\\result\\' + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-')) + FLAGS.scope
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    print_time()
    tf.reset_default_graph()

    # Model Code Block
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, '..\\data_rand\\clause_keywords.csv', FLAGS.w2v_file)
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    print('build model...')
    # CNN filter sizes
    filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))

    # Model placeholder definition
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    ll = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_label])
    el = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_elabel])
    placeholders = [x, sen_len, doc_len, keep_prob1, keep_prob2, y, ll, el]

    # Build model and state the loss function
    loss, pred_class, reg = build_model(word_embedding, x, sen_len, doc_len, keep_prob1, keep_prob2, y, ll, el, filter_sizes)
    loss_op = loss * FLAGS.cause + reg * FLAGS.l2_reg
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)
    true_y_op = y

    print('build model done!\n')

    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        acc_cause_list, p_cause_list, r_cause_list, f1_cause_list = [], [], [], []
        acc_pos_list, p_pos_list, r_pos_list, f1_pos_list = [], [], [], []
        p_pair_list, r_pair_list, f1_pair_list = [], [], []

        # To avoid randomness, perform 10 times experiments with each fold of dataset being the testing set
        for fold in range(1, 11):
            sess.run(tf.global_variables_initializer())
            # train for one fold
            logger.info('############# fold {} begin ###############'.format(fold))

            # Data Code Block
            train_file_name = 'fold{}_train.txt'.format(fold)
            test_file_name = 'fold{}_test.txt'.format(fold)
            _, tr_x, tr_sen_len, tr_doc_len, tr_y, tr_l, tr_el = load_data('..\\data_rand\\'+train_file_name, word_id_mapping, label2id, emo2id, FLAGS.max_doc_len, FLAGS.max_sen_len)
            te_doc_id, te_x, te_sen_len, te_doc_len, te_y, te_l, te_el = load_data('..\\data_rand\\'+test_file_name, word_id_mapping, label2id, emo2id, FLAGS.max_doc_len, FLAGS.max_sen_len)

            max_f1_cause, max_f1_pos, max_f1_avg = [-1.] * 3
            logger.info('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))

            for i in range(FLAGS.training_iter):
                start_time, step = time.time(), 1
                # train
                for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, FLAGS.keep_prob1, FLAGS.keep_prob2, tr_y, tr_l, tr_el, FLAGS.batch_size):
                    _, loss, pred_y, true_y, doc_len_batch = sess.run(
                        [optimizer, loss_op, pred_class, true_y_op, doc_len], feed_dict=dict(zip(placeholders, train)))
                    if step % 10 == 0:
                        logger.info('step {}: train loss {:.4f} '.format(step, loss))

                        # evaluation metrics calculation
                        pred_y_cause, true_y_cause, pred_y_pos, true_y_pos, pp, pr, pf1 = mapcausepos(pred_y, true_y)

                        logger.info('pair_predict: train p {:.4f} r {:.4f} f1 {:.4f}'.format(pp, pr, pf1))
                    step = step + 1

                # test
                test = [te_x, te_sen_len, te_doc_len, 1., 1., te_y, te_l, te_el]
                loss, pred_y, true_y, doc_len_batch = sess.run(
                        [loss_op, pred_class, true_y_op, doc_len], feed_dict=dict(zip(placeholders, test)))
                logger.info('\nepoch {}: test loss {:.4f} cost time: {:.1f}s\n'.format(i, loss, time.time()-start_time))

                pred_y_cause, true_y_cause, pred_y_pos, true_y_pos, pp, pr, pf1 = mapcausepos(pred_y, true_y)
                logger.info('pair_predict: test p {:.4f} r {:.4f} f1 {:.4f}'.format(pp, pr, pf1))

                acc, p, r, f1 = acc_prf(pred_y_cause, true_y_cause, doc_len_batch)
                result_avg_cause = [acc, p, r, f1]

                acc, p, r, f1 = acc_prf(pred_y_pos, true_y_pos, doc_len_batch)
                result_avg_pos = [acc, p, r, f1]

                # Judge whether we need to update the results
                if pf1 > max_f1_avg:
                    max_f1_avg = pf1
                    result_pair_max = [pp, pr, pf1]
                    result_cause_max = result_avg_cause
                    result_pos_max = result_avg_pos

                    logger.info('Best result updated in Iteration {}'.format(i))
                    tmp_file = open(res_dir + "/best_pred_for_fold_{}.txt".format(fold), 'w')
                    true_y = np.argmax(true_y, 2)
                    pred_y = np.argmax(pred_y, 2) if len(pred_y.shape) > 2 else pred_y
                    for g in range(len(te_doc_id)):
                        tmp_file.write(str(te_doc_id[g]))
                        tmp_file.write(':\n')
                        for j in range(te_doc_len[g]):
                            tmp_file.write(str(j))
                            tmp_file.write(',')
                            tmp_file.write(str(true_y[g][j]))
                            tmp_file.write(',')
                            tmp_file.write(str(pred_y[g][j]))
                            tmp_file.write('\n')
                        tmp_file.write('\n')
                    tmp_file.close()

                logger.info('Best pair: max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(result_pair_max[0], result_pair_max[1], result_pair_max[2]))
                logger.info('Average max cause: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(result_cause_max[0], result_cause_max[1], result_cause_max[2], result_cause_max[3]))
                logger.info('Average max pos: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(result_pos_max[0], result_pos_max[1], result_pos_max[2], result_pos_max[3]))

            logger.info('Optimization Finished!\n')
            logger.info('############# fold {} end ###############'.format(fold))

            # Record all results for the current fold
            acc_cause_list.append(result_cause_max[0])
            p_cause_list.append(result_cause_max[1])
            r_cause_list.append(result_cause_max[2])
            f1_cause_list.append(result_cause_max[3])
            acc_pos_list.append(result_pos_max[0])
            p_pos_list.append(result_pos_max[1])
            r_pos_list.append(result_pos_max[2])
            f1_pos_list.append(result_pos_max[3])
            p_pair_list.append(result_pair_max[0])
            r_pair_list.append(result_pair_max[1])
            f1_pair_list.append(result_pair_max[2])

        print_training_info()
        all_results = [acc_cause_list, p_cause_list, r_cause_list, f1_cause_list, acc_pos_list, p_pos_list, r_pos_list, f1_pos_list, p_pair_list, r_pair_list, f1_pair_list]
        acc_cause, p_cause, r_cause, f1_cause, acc_pos, p_pos, r_pos, f1_pos, p_pair, r_pair, f1_pair = map(lambda x: np.array(x).mean(), all_results)
        logger.info('\ncause_predict: test f1 in 10 fold: {}'.format(np.array(f1_cause_list).reshape(-1, 1)))
        logger.info('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_cause, p_cause, r_cause, f1_cause))
        logger.info('position_predict: test f1 in 10 fold: {}'.format(np.array(f1_pos_list).reshape(-1, 1)))
        logger.info('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_pos, p_pos, r_pos, f1_pos))
        logger.info('pair_predict: test f1 in 10 fold: {}'.format(np.array(f1_pair_list).reshape(-1, 1)))
        logger.info('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p_pair, r_pair, f1_pair))
        print_time()

    loggercloser(logger)


def main(_):
    FLAGS.num_layers = 1
    FLAGS.softmax_or_crf = 1
    FLAGS.scope = '1trans_softmax'
    run()

    FLAGS.softmax_or_crf = 0
    FLAGS.scope = '1trans_crf'
    run()

    FLAGS.num_layers = 2
    FLAGS.softmax_or_crf = 1
    FLAGS.scope = '2trans_softmax'
    run()

    FLAGS.softmax_or_crf = 0
    FLAGS.scope = '2trans_crf'
    run()


if __name__ == '__main__':
    tf.app.run()
