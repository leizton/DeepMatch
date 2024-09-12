import os
import sys
import random
import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import save_model, load_model

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from deepmatch.layers import custom_objects


def gen_data_set(data, max_seq_len=50, negsample=5, test_ratio=0.1):
    data.sort_values('timestamp', inplace=True)
    train_set, test_set = [], []
    for user_id, user_history in tqdm.tqdm(data.groupby('user_id')):
        pos_list = user_history['item_id'].tolist()
        category_list = user_history['categories'].tolist()
        rating_list = user_history['rating'].tolist()
        neg_rating = (np.sum(rating_list) / len(rating_list)) * 0.4
        if len(pos_list) < 30:
            continue
        if negsample > 0:
          item_ids = data['item_id'].unique()
          candidate_set = list(set(item_ids) - set(pos_list))
          neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        test_set_start_index = max(1, int(len(pos_list) * (1 - test_ratio)))
        for i in range(1, len(pos_list)):
            seq_len = min(i, max_seq_len)
            item_seq = pos_list[:i][::-1][:seq_len]
            item_category_seq = category_list[:i][::-1][:seq_len]
            pos_sample = (user_id, pos_list[i], 1, item_seq, seq_len, item_category_seq, rating_list[i])
            if i < test_set_start_index:
                # 正样本
                train_set.append(pos_sample)
                # 负样本
                for neg_i in range(negsample):
                    neg_sample = (user_id, neg_list[i*negsample+neg_i], 0, item_seq, seq_len, item_category_seq, neg_rating)
                    train_set.append(neg_sample)
            else:
                test_set.append(pos_sample)
    random.shuffle(train_set)
    random.shuffle(test_set)
    return train_set, test_set


def gen_model_input(train_set, user_table, max_seq_len):
    train_user_id = np.array([line[0] for line in train_set])
    train_item_id = np.array([line[1] for line in train_set])
    train_label = np.array([line[2] for line in train_set])
    train_item_seq = [line[3] for line in train_set]
    train_hist_len = np.array([line[4] for line in train_set])
    train_item_category_seq = [line[5] for line in train_set]
    train_rating = np.array([line[6] for line in train_set])
    train_item_seq_pad = pad_sequences(train_item_seq, maxlen=max_seq_len, padding='post', truncating='post', value=0)
    train_item_category_seq_pad = pad_sequences(train_item_category_seq, maxlen=max_seq_len, padding='post', truncating='post', value=0)
    train_model_input = {
        'user_id': train_user_id,
        'item_id': train_item_id,
        'hist_movie_id': train_item_seq_pad,
        'hist_categories': train_item_category_seq_pad,
        'hist_len': train_hist_len,
        'rating': train_rating,
    }
    for key in ['gender', 'age', 'occupation', 'zipcode']:
        train_model_input[key] = user_table.loc[train_model_input['user_id']][key].values
    return train_model_input, train_label


def calc_emb_sim(v1, v2):
    return (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1) * 0.5


if __name__ == '__main__':
    negsample = 3
    SEQ_LEN = 50  # 序列最大长度
    embedding_dim = 64

    # Load data
    # user_id,item_id,rating,timestamp,title,categories,gender,age,occupation,zipcode
    data_file = project_dir + '/examples/movielens_sample.txt'
    data = pd.read_csv(data_file)
    data['categories'] = list(map(lambda x: x.split('|')[0], data['categories'].values))
    data['rating'] = data['rating'].astype(float)
    data['rating'] = (data['rating'] - np.min(data['rating'])) / (np.max(data['rating']) - np.min(data['rating']))

    # Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`
    feature_max_idx = {}
    sparse_features = ['item_id', 'user_id', 'gender', 'age', 'occupation', 'zipcode', 'categories']
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        print('feature_max_idx %s %s' % (feature, feature_max_idx[feature] - 1))

    user_table = data[['user_id', 'gender', 'age', 'occupation', 'zipcode']].drop_duplicates('user_id')
    user_table.set_index('user_id', inplace=True)
    item_table = data[['item_id']].drop_duplicates('item_id')
    item_table.sort_values('item_id', inplace=True)

    train_set, test_set = gen_data_set(data, SEQ_LEN, negsample)
    train_model_input, train_label = gen_model_input(train_set, user_table, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_table, SEQ_LEN)
    print('train_size: %s, test_size: %s' % (len(train_set), len(test_set)))

    # Count unique features for each sparse field and generate feature config for sequence feature
    user_feature_columns = [
        SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
        SparseFeat('gender', feature_max_idx['gender'], embedding_dim),
        SparseFeat('age', feature_max_idx['age'], embedding_dim),
        SparseFeat('occupation', feature_max_idx['occupation'], embedding_dim),
        SparseFeat('zipcode', feature_max_idx['zipcode'], embedding_dim),
        VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['item_id'], embedding_dim, embedding_name='item_id'), SEQ_LEN, 'mean', 'hist_len'),
        VarLenSparseFeat(SparseFeat('hist_categories', feature_max_idx['categories'], embedding_dim, embedding_name='categories'), SEQ_LEN, 'mean', 'hist_len')
    ]
    item_feature_columns = [SparseFeat('item_id', feature_max_idx['item_id'], embedding_dim)]
    item_train_counter = Counter(train_model_input['item_id'])
    item_count = [item_train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('frequency', num_sampled=5, item_name='item_id', item_count=item_count)

    # Define Model and train
    tf.compat.v1.disable_eager_execution()
    model = YoutubeDNN(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, embedding_dim), sampler_config=sampler_config)
    model.compile(optimizer='adam', loss=sampledsoftmaxloss)
    model.fit(train_model_input, train_label, batch_size=256, epochs=10, verbose=1, validation_split=0)

    # Generate user features for testing and full item features for retrieval
    all_item_model_input = {'item_id': item_table['item_id'].values}
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2**12)

    # Export model
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    model_h5_file = project_dir + '/y2bdnn.h5'
    save_model(user_embedding_model, model_h5_file)
    print('save_model to %s' % model_h5_file)

    # Predict
    user_embedding_model = load_model(model_h5_file, custom_objects)
    print('load_model from %s' % model_h5_file)
    user_embs = user_embedding_model.predict(test_model_input, batch_size=2**12)
    loss_test, loss_baseline = 0., 0.
    test_num = len(user_embs)
    for i in range(test_num):
        item_emb = item_embs[test_model_input['item_id'][i]]
        rating_test = calc_emb_sim(user_embs[i], item_emb)
        rating_baseline = 0.5
        rating_true = test_model_input['rating'][i]
        loss_test += abs(rating_test - rating_true)
        loss_baseline += abs(rating_baseline - rating_true)
    loss_test, loss_baseline = loss_test/test_num, loss_baseline/test_num
    print('loss_test: %s, loss_baseline: %s' % (loss_test, loss_baseline))
