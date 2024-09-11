import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from preprocess import gen_data_set, gen_model_input


if __name__ == '__main__':
    # user_id,movie_id,rating,timestamp,title,genres,gender,age,occupation,zip
    movielens_file = project_dir + '/examples/movielens_sample.txt'
    data = pd.read_csvdata = pd.read_csv(movielens_file)
    data['genres'] = list(map(lambda x: x.split('|')[0], data['genres'].values))
    sparse_features = ['movie_id', 'user_id', 'gender', 'age', 'occupation', 'zip', 'genres']
    SEQ_LEN = 50

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`
    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        print('feature_max_idx %s %s' % (feature, feature_max_idx[feature]))

    user_profile = data[['user_id', 'gender', 'age', 'occupation', 'zip']].drop_duplicates('user_id')

    item_profile = data[['movie_id']].drop_duplicates('movie_id')

    user_profile.set_index('user_id', inplace=True)

    user_item_list = data.groupby('user_id')['movie_id'].apply(list)

    train_set, test_set = gen_data_set(data, SEQ_LEN, 0)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)
    for e in train_model_input:
        print('train_model_input key: %s' % e)
    print('train_label sum: %s' % np.sum(train_label))
    print('test_label sum: %s' % np.sum(test_label))

    # 2.count #unique features for each sparse field and generate feature config for sequence feature
    embedding_dim = 16
    user_feature_columns = [
        SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
        SparseFeat('gender', feature_max_idx['gender'], embedding_dim),
        SparseFeat('age', feature_max_idx['age'], embedding_dim),
        SparseFeat('occupation', feature_max_idx['occupation'], embedding_dim),
        SparseFeat('zip', feature_max_idx['zip'], embedding_dim),
        VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim, embedding_name='movie_id'), SEQ_LEN, 'mean', 'hist_len'),
        VarLenSparseFeat(SparseFeat('hist_genres', feature_max_idx['genres'], embedding_dim, embedding_name='genres'), SEQ_LEN, 'mean', 'hist_len')
    ]
    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]
    train_counter = Counter(train_model_input['movie_id'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('frequency', num_sampled=5, item_name='movie_id', item_count=item_count)

    # 3.Define Model and train
    tf.compat.v1.disable_eager_execution()
    model = YoutubeDNN(
        user_feature_columns, item_feature_columns,
        user_dnn_hidden_units=(64, embedding_dim),
        sampler_config=sampler_config)
    model.compile(optimizer='adam', loss=sampledsoftmaxloss)
    history = model.fit(
        train_model_input, train_label,  # train_label,
        batch_size=256, epochs=1, verbose=1, validation_split=0.0, )

    # 4. Generate user features for testing and full item features for retrieval
    test_user_model_input = test_model_input
    all_item_model_input = {'movie_id': item_profile['movie_id'].values}
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)
    print(user_embs.shape)
    print(item_embs.shape)
