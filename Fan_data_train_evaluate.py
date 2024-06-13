import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse

from tensorflow.keras.layers import Dense, Input, Multiply, Lambda, \
    Embedding, Activation, GlobalMaxPooling1D, Embedding, \
    Dropout, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import os
from Utils import read_and_load_data_tvt_combine, create_data_from_score, isNaN

tf.random.set_seed(10086)
np.random.seed(10086)


class SampleConcrete(tf.keras.layers.Layer):
    def __init__(self, args):
        super(SampleConcrete, self).__init__()
        self.args = args

    def call(self, logits):
        lo_gits_ = K.permute_dimensions(logits, (0, 2, 1))

        uni_shape = K.shape(logits)[0]
        uniform_a = K.random_uniform(shape=(uni_shape, 1, self.args.sents),
                                     minval=np.finfo(tf.float32.as_numpy_dtype).tiny, maxval=1.0)
        uniform_b = K.random_uniform(shape=(uni_shape, 1, self.args.sents),
                                     minval=np.finfo(tf.float32.as_numpy_dtype).tiny, maxval=1.0)

        gumbel_a = -K.log(-K.log(uniform_a))
        gumbel_b = -K.log(-K.log(uniform_b))

        no_z_lo_gits = K.exp((gumbel_a + lo_gits_) / self.args.tau)
        de_z_lo_gits = no_z_lo_gits + K.exp((gumbel_b + (1.0 - lo_gits_)) / self.args.tau)

        samples = no_z_lo_gits / de_z_lo_gits

        logits = tf.reshape(lo_gits_, [-1, lo_gits_.shape[-1]])
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.args.selected_sents, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(K.permute_dimensions(samples, (0, 2, 1)), tf.expand_dims(discrete_logits, -1))


def construct_gumbel_selector(review_input, args, vocab_size):
    sentence_input = Input(shape=(args.sent_len,), dtype='int32')
    embedding_layer = Embedding(vocab_size,
                                args.embedding_dim,
                                input_length=args.sent_len,
                                name='embedding',
                                trainable=True)

    embedded_sequences = embedding_layer(sentence_input)
    net = Dropout(args.drop_rate)(embedded_sequences)
    '''
    net = Conv1D(250, 3,
                 padding='valid',
                 activation='relu',
                 strides=1)(net)'''
    net = GlobalMaxPooling1D()(net)
    sent_encoder = Model(sentence_input, net)

    review_encoder = TimeDistributed(sent_encoder)(review_input)
    net = review_encoder

    net_1 = Dropout(args.drop_rate, name='dropout_1')(net)
    x_net_1 = Dense(args.dim_dnn, name='new_dense_1', activation='relu')(net_1)

    net_2 = Dropout(args.drop_rate, name='dropout_2')(x_net_1)
    x_net_2 = Dense(args.dim_dnn, name='new_dense_2', activation='relu')(net_2)

    net_logits = Dropout(args.drop_rate, name='dropout_logits')(x_net_2)

    x_net_logits = Dense(1, name='new_dense_logits', activation='sigmoid')(net_logits)

    return x_net_logits


Sum = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda x: [x[0], x[2]])


def selection_process(args, vocab_size):
    x_input = Input(shape=(args.sents, args.sent_len), dtype='int32')
    logits_T = construct_gumbel_selector(x_input, args, vocab_size)
    T = SampleConcrete(args)(logits_T)

    sentence_input = Input(shape=(args.sent_len,), dtype='int32')

    embedding_layer = Embedding(vocab_size,
                                args.embedding_dim,
                                input_length=args.sent_len,
                                name='embedding',
                                trainable=True)
    embedded_sequences = embedding_layer(sentence_input)
    net = Dropout(args.drop_rate)(embedded_sequences)
    '''
    net = Conv1D(250, 3,
                  padding='valid',
                  activation='relu',
                  strides=1)(net)'''
    net = GlobalMaxPooling1D()(net)
    sent_encoder2 = Model(sentence_input, net)
    review_encoder2 = TimeDistributed(sent_encoder2)(x_input)

    selected_encoding = Multiply()([review_encoder2, T])

    model = Model(inputs=x_input, outputs=[selected_encoding, logits_T, review_encoder2])
    return model


def classifier_process(args):
    x_input = Input(shape=(args.sents, 150), dtype='float32')
    net = Sum(x_input)
    net = Dense(250)(net)
    net = Activation('relu')(net)
    predicts = Dense(2, activation='softmax', name='new_dense')(net)

    model = Model(inputs=x_input, outputs=predicts)
    return model


def contrastive_loss(logits, labels, review_encoder, args):
    logits_p = K.permute_dimensions(logits, (0, 2, 1))
    logits = tf.reshape(logits_p, [-1, logits_p.shape[-1]])
    threshold = tf.expand_dims(tf.nn.top_k(logits, args.selected_sents, sorted=True)[0][:, -1], -1)
    discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)
    discrete_logits = tf.expand_dims(discrete_logits, -1)

    selected_encoding_f = Multiply()([review_encoder, discrete_logits])

    selected_rs = tf.reshape(selected_encoding_f, [-1, args.sents * args.embedding_dim])
    scaler = StandardScaler()
    scaled_selected_rs = scaler.fit_transform(selected_rs)
    kmeans = KMeans(init="random", n_clusters=args.clusters, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(scaled_selected_rs)

    cluster_labels = kmeans.labels_ + 1
    cluster_labels = cluster_labels * labels

    selected_rst = tf.transpose(selected_rs)

    selected_encodes_uv = tf.matmul(selected_rs, selected_rst)
    selected_encodes_u_l2 = tf.sqrt(tf.reduce_sum(tf.pow(selected_rs, 2), axis=-1, keepdims=True))
    selected_encodes_v_l2 = tf.transpose(tf.sqrt(tf.reduce_sum(tf.pow(selected_rs, 2), axis=-1, keepdims=True)))
    selected_encodes_cosine = Multiply()([selected_encodes_uv, 1.0/(tf.matmul(selected_encodes_u_l2, selected_encodes_v_l2))])

    matrix_dot_lts = tf.math.divide(selected_encodes_cosine, args.temp)

    y_train_c = np.int32(cluster_labels)
    y_train_c = np.reshape(y_train_c, [y_train_c.shape[0], 1])
    y_train_ct = np.transpose(y_train_c)
    mask_p = tf.equal(y_train_c, y_train_ct)
    mask_pf = tf.cast(mask_p, tf.float32)

    cardinality_p = tf.math.reduce_sum(mask_pf, axis=-1, keepdims=True)

    mask_a = tf.cast(tf.eye(y_train_c.shape[0]), tf.float32)
    mask_a = 1.0 - mask_a

    exp_matrix_dot = tf.math.exp(matrix_dot_lts)

    matrix_a = Multiply()([exp_matrix_dot, mask_a])

    matrix_a_sum = tf.math.reduce_sum(matrix_a, axis=-1, keepdims=True)

    log_prob = tf.math.log(matrix_dot_lts + 1e-10) - tf.math.log(matrix_a_sum + 1e-10)

    modified_cl = tf.math.reduce_sum(Multiply()([log_prob, mask_pf]), axis=-1, keepdims=True) * labels
    mean_log_prob_p = tf.math.reduce_mean(modified_cl / cardinality_p)

    return mean_log_prob_p


def LiVuITCL_train(args, x_train, x_val, y_train, y_val, vocab_size):
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    opt = optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    print('Creating model...')
    selection_network = selection_process(args, vocab_size)
    classifier_network = classifier_process(args)

    saved_model_dir = args.home_dir + 'saved_models' + '/' + str(args.lr) + '_' + str(args.sigma) + '_' + str(args.tau) + '_' + str(args.temp) + '_' + str(args.dim_dnn) + '_' + str(args.clusters) + '/'
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)

    results_file = open(args.home_dir + 'results.txt', 'a+')
    results_file.write('\np_lr: %f --- p_sigma: %f --- p_tau: %f --- p_temp: %f --- p_dnn: %f --- p_clusters: %f \n' % (args.lr, args.sigma, args.tau, args.temp, args.dim_dnn, args.clusters))
    
    val_acc_max = 0.0

    for epoch in range(args.train_epochs):
        results_file.write("epoch: %d \n" % epoch)
        print("epoch: %d" % epoch)
        for batch_idx, (data, labels) in enumerate(train_data):
            with tf.GradientTape() as tape:
                selected_encodes, logits_p, review_encoder_f = selection_network(data, training=True)
                predicts = classifier_network(selected_encodes, training=True)

                loss = loss_fn(labels, predicts)
                mean_log_prob_p = contrastive_loss(logits_p, labels, review_encoder_f, args)
                if isNaN(mean_log_prob_p):
                    mean_log_prob_p = 0.0
                
                total_loss = loss - args.sigma*mean_log_prob_p

                variables = selection_network.trainable_variables + classifier_network.trainable_variables
                gradients = tape.gradient(total_loss, variables)
                opt.apply_gradients(zip(gradients, variables))

                train_acc_metric.update_state(labels, predicts)

                if batch_idx % 10 == 0:
                    results_file.write("Training loss at step %d -- loss: %.4f -- mean_log_prob_p: %.4f \n" % (
                        batch_idx, float(loss), float(mean_log_prob_p)))
                    results_file.write("Seen so far: %d samples \n" % ((batch_idx + 1) * args.batch_size))
                    print("Training loss at step %d -- loss: %.4f -- mean_log_prob_p: %.4f \n" % (
                        batch_idx, float(loss), float(mean_log_prob_p)))
                    print("Seen so far: %d samples" % ((batch_idx + 1) * args.batch_size))

        train_acc = train_acc_metric.result()
        results_file.write("Training acc over epoch: %.4f \n" % (float(train_acc)))
        print("Training acc over epoch: %.4f" % (float(train_acc)))
        train_acc_metric.reset_states()

        for x_batch_val, y_batch_val in valid_data:
            selected_encodes, _, _ = selection_network(x_batch_val, training=False)
            val_logits = classifier_network(selected_encodes, training=False)
            
            val_acc_metric.update_state(y_batch_val, val_logits)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        results_file.write("Validation acc: %.4f \n" % (float(val_acc)))
        print("Validation acc: %.4f" % (float(val_acc)))

        if val_acc_max < val_acc:
            val_acc_max = val_acc
            print('saving a model')
            selection_network.save(saved_model_dir + 'selection_network_model')
            classifier_network.save(saved_model_dir + 'classifier_network_model')

    results_file.close()


def LiVuITCL_eval(args, history_file, x_val, y_val, gts_val, word_index):
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    valid_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)

    saved_model_dir = args.home_dir + 'saved_models' + '/' + str(args.lr) + '_' + str(args.sigma) + '_' + str(args.tau) + '_' + str(args.temp) + '_' + str(args.dim_dnn) + '_' + str(args.clusters) + '/'
    if not os.path.exists(saved_model_dir):
        print('cannot find out the saved models')

    history_file.write('p_lr: %f --- p_sigma: %f --- p_tau: %f --- p_temp: %f --- p_dnn: %f --- p_clusters: %f \n' % (args.lr, args.sigma, args.tau, args.temp, args.dim_dnn, args.clusters))

    selection_network = keras.models.load_model(saved_model_dir + 'selection_network_model')
    classifier_network = keras.models.load_model(saved_model_dir + 'classifier_network_model')

    val_logits_ls = np.array([])
    val_y_list = np.array([])

    val_p_ls = np.array([])
    val_x_list = np.array([])

    for x_batch_val, y_batch_val in valid_data:
        selected_encodes, logits_p, _ = selection_network(x_batch_val, training=False)
        val_logits = classifier_network(selected_encodes, training=False)

        val_logits_ls = np.append(val_logits_ls, val_logits)
        val_y_list = np.append(val_y_list, y_batch_val)

        val_p_ls = np.append(val_p_ls, logits_p)
        val_x_list = np.append(val_x_list, x_batch_val)

        val_acc_metric.update_state(y_batch_val, val_logits)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()

    history_file.write("Validation acc: %.4f \n" % (float(val_acc)))
    print("Validation acc: %.4f" % (float(val_acc)))

    return saved_model_dir, val_p_ls, val_logits_ls, val_x_list, val_y_list, gts_val, word_index


def main():
    """
    ===for the training process===
    for example,
    python Fan_data_train_evaluate.py --lr=1e-4 --sigma=1e-1 --tau=0.5 --temp=0.5 --dim_dnn=300 --clusters=7 --train_epochs=5 --home_dir=./Fan_data_results/ --do_train
    ===for the evaluation process===
    for example,
    python Fan_data_train_evaluate.py --lr=1e-4 --sigma=1e-1 --tau=0.5 --temp=0.5 --dim_dnn=300 --clusters=7 --home_dir=./Fan_data_results/ --do_eval
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=1e-4, type=float, help="The learning rate used in the training process.")
    parser.add_argument("--sigma", default=1e-1, type=float, help="Sigma is the trade-off hyper-parameter for the cluster term.")
    parser.add_argument("--tau", default=0.5, type=float, help="Tau value used in the Gumbel Softmax.")

    parser.add_argument("--temp", default=0.5, type=float, help="Temperature value used in the Constrative learning term.")
    parser.add_argument("--dim_dnn", default=300, type=int, help="The dimension of the dense layer used in the model.")
    parser.add_argument("--clusters", default=7, type=int, help="The number of clusters used in the cluster spatial contrastive learning term.")

    parser.add_argument("--train_epochs", default=5, type=int, help="The number of epochs used to train the model.")
    parser.add_argument("--selected_sents", default=10, type=int, help="The number of topK selected statements.")

    parser.add_argument("--sent_len", default=15, type=int, help="The length of each statement.")
    parser.add_argument("--sents", default=150, type=int, help="The length of each function.")
    parser.add_argument("--embedding_dim", default=150, type=int, help="The embedding dimension.")
    parser.add_argument("--batch_size", default=128, type=int, help="The batch size of data.")
    parser.add_argument("--drop_rate", default=0.2, type=float, help="The dropout rate used in the the Dropout layer.")

    parser.add_argument("--home_dir", default='./Fan_data_results/', type=str, help="The home directory used to save all the results.")

    parser.add_argument("--do_train", action='store_true', help="Whether to run the training process.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run the evaluation process.")

    args = parser.parse_args()

    print('Loading data set...')
    data_set = read_and_load_data_tvt_combine(args.sents, args.sent_len)

    x_train, x_val, y_train, y_val = data_set['x_train'], data_set['x_valid'], data_set['y_train'], data_set['y_valid']
    y_train = np.float32(y_train)

    gts_val = data_set['gts_valid']
    word_index, vocab_size = data_set['word_index'], data_set['vocabulary_size'][0][0]

    if args.do_train:
        LiVuITCL_train(args, x_train, x_val, y_train, y_val, vocab_size)
    if args.do_eval:
        result_dir = args.home_dir + 'history_logs_load_predictions' + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        history_file = open(result_dir + 'his_load_predictions.txt', 'a+')

        saved_model_dir, logits_p, val_logits, x_val, y_val, gts_val, word_index = LiVuITCL_eval(args, history_file, x_val, y_val, gts_val, word_index)
        print('Creating data set with selected sentences...')
        val_logits = np.argmax(np.reshape(val_logits, [-1, 2]), axis=1)
        x_val = np.reshape(x_val, [-1, args.sents, args.sent_len])
        logits_p = np.reshape(logits_p, [-1, args.sents])

        create_data_from_score(saved_model_dir, x_val, logits_p, y_val, val_logits, gts_val, word_index)
        history_file.close()


if __name__ == "__main__":
    main()
