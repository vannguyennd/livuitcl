from __future__ import print_function
import numpy as np
import pickle as pkl
import scipy.io as io
import tensorflow as tf
import os

def read_and_load_data_tvt_combine(max_statements, max_length_statements):
    dir_path =  'Fan_dataset/'

    with open(dir_path + 'ind_word.pickle', 'rb') as f:
        word_index = pkl.load(f)

    ind_word_vocab_size_path = dir_path + 'ind_word_vocab_size' + '.mat'
    ind_word_vocab_size = io.loadmat(ind_word_vocab_size_path)
    vocabulary_size = ind_word_vocab_size['vocab_size']

    data_non_path = dir_path + 'data_non' + '.mat'
    data_non = io.loadmat(data_non_path)
    X_non = data_non['X_non']
    X_non_len = data_non['X_non_len']
    X_non_mask = np.reshape(data_non['X_non_mask'], [-1])
    y_non = np.reshape(data_non['y_non'], [-1])
    gts_non = np.reshape(data_non['gts_non'], [-1])

    data_vul_path = dir_path + 'data_vul' + '.mat'
    data_vul = io.loadmat(data_vul_path)
    X_vul = data_vul['X_vul']
    X_vul_len = data_vul['X_vul_len']
    X_vul_mask = np.reshape(data_vul['X_vul_mask'], [-1])
    y_vul = np.reshape(data_vul['y_vul'], [-1])
    gts_vul = np.reshape(data_vul['gts_vul'], [-1])

    percent = 0.8
    non_size = int(X_non.shape[0] * percent)
    vul_size = int(X_vul.shape[0] * percent)

    X_non, X_non_len, X_non_mask, y_non, gts_non = shuffle_aligned_list([X_non, X_non_len, X_non_mask, y_non, gts_non])
    X_vul, X_vul_len, X_vul_mask, y_vul, gts_vul = shuffle_aligned_list([X_vul, X_vul_len, X_vul_mask, y_vul, gts_vul])

    X_train = np.concatenate((X_non[:non_size], X_vul[:vul_size]))
    X_len_train = np.concatenate((X_non_len[:non_size], X_vul_len[:vul_size]))
    X_mask_train = np.concatenate((X_non_mask[:non_size], X_vul_mask[:vul_size]))
    y_train = np.concatenate((y_non[:non_size], y_vul[:vul_size]))
    gts_train = np.concatenate((gts_non[:non_size], gts_vul[:vul_size]))

    X_train, X_len_train, X_mask_train, y_train, gts_train = shuffle_aligned_list([X_train, X_len_train, X_mask_train, y_train, gts_train])

    X_valid = np.concatenate((X_non[non_size:], X_vul[vul_size:]))
    X_len_valid = np.concatenate((X_non_len[non_size:], X_vul_len[vul_size:]))
    X_mask_valid = np.concatenate((X_non_mask[non_size:], X_vul_mask[vul_size:]))
    y_valid = np.concatenate((y_non[non_size:], y_vul[vul_size:]))
    gts_valid = np.concatenate((gts_non[non_size:], gts_vul[vul_size:]))

    X_valid, X_len_valid, X_mask_valid, y_valid, gts_valid = shuffle_aligned_list([X_valid, X_len_valid, X_mask_valid, y_valid, gts_valid])

    data_set = {'x_train': X_train[:,:max_statements,:max_length_statements], 'y_train': y_train, 'gts_train': gts_train,
                'x_valid': X_valid[:,:max_statements,:max_length_statements], 'y_valid': y_valid, 'gts_valid': gts_valid,
                'word_index': word_index, 'vocabulary_size': vocabulary_size}
    print('Data loaded...')
    return data_set


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    np.random.seed(123)
    p = np.random.permutation(num)
    return [d[p] for d in data]


def isNaN(value):
    return value != value


def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def get_selected_sent_s(score):
    selected = np.argsort(score)[-score.shape[0]:]
    x_selected = selected
    return x_selected


def convert_into_word(i_data, id_to_word):
    results = ''
    for word_id in i_data:
        results += id_to_word[word_id] + ' '
    results += '\n'
    return results


def check_exist_in(item_list, list_item):
    for i in range(len(item_list)):
        if int(item_list[i]) != int(list_item[i]):
            return False
    return True


def create_data_from_score(saved_dir, x_test_res, scores, y_te_record, y_t_pre, gts_te, id_to_word, mx_step=10):
    if len(scores.shape) == 3:
        scores = np.squeeze(scores)

    x_selected_idx_list = []
    for i in range(scores.shape[0]):
        x_selected_idx = get_selected_sent_s(scores[i])
        x_selected_idx_list.append(np.flip(x_selected_idx, 0))

    x_new_indicies = np.array(x_selected_idx_list)
    x_test_res = np.array(x_test_res)

    vul_result = saved_dir + 'vul_result_' + str(mx_step) + '/'
    if not os.path.exists(vul_result):
        os.makedirs(vul_result)

    non_result = saved_dir + 'non_result_' + str(mx_step) + '/'
    if not os.path.exists(non_result):
        os.makedirs(non_result)

    for i_batch in range(x_test_res.shape[0]):
        if y_te_record[i_batch] == 1:
            record_result = open(vul_result + '/' + str(i_batch) + '.txt', "w")
        else:
            record_result = open(non_result + '/' + str(i_batch) + '.txt', "w")

        source_all = 'true: ' + str(y_te_record[i_batch]) + ' vs predict: ' + str(y_t_pre[i_batch]) + '\n'
        source_all += '-----following are original function-----\n'
        source_select_ordered = ''

        for j_sent in range(x_test_res[i_batch].shape[0]):
            source_all += convert_into_word(x_test_res[i_batch][j_sent], id_to_word)
        for idx_line in x_new_indicies[i_batch]:
            source_select_ordered += convert_into_word(x_test_res[i_batch][idx_line], id_to_word)

        source_all += '-----followings are all selected statements in order-----\n'
        source_all += source_select_ordered

        ground_truth = str(gts_te[i_batch])
        source_all += '-----followings are ground truth-----\n'
        source_all += ground_truth
        record_result.write(source_all)

        record_result.close()
