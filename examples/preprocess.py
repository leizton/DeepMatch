import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


def gen_data_set(data, seq_max_len=50, negsample=5):
    negsample = max(1, negsample)
    data.sort_values('timestamp', inplace=True)
    item_ids = data['movie_id'].unique()
    item_id_genres_map = dict(zip(data['movie_id'].values, data['genres'].values))
    train_set, test_set = [], []
    for user_id, user_history in tqdm(data.groupby('user_id')):
        pos_list = user_history['movie_id'].tolist()
        genres_list = user_history['genres'].tolist()
        rating_list = user_history['rating'].tolist()
        if len(pos_list) < 30:
            continue
        candidate_set = list(set(item_ids) - set(pos_list))
        neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        test_set_start_index = max(1, int(len(pos_list) * 0.9))
        for i in range(1, len(pos_list)):
            seq_len = min(i, seq_max_len)
            item_history = pos_list[:i][::-1][:seq_len]
            item_genres_history = genres_list[:i][::-1][:seq_len]
            # 第1个是正样本, 跟着negsample个负样本
            pos_sample = (user_id, pos_list[i], 1, item_history, seq_len, item_genres_history, genres_list[i], rating_list[i])
            if i < test_set_start_index:
                train_set.append(pos_sample)
                for neg_i in range(negsample):
                    neg_item_id = neg_list[i * negsample + neg_i]
                    neg_sample = (user_id, neg_item_id, 0, item_history, seq_len, item_genres_history, item_id_genres_map[neg_item_id], 1)
                    train_set.append(neg_sample)
            else:
                test_set.append(pos_sample)
    random.shuffle(train_set)
    random.shuffle(test_set)
    print('train_size=%s, test_size=%s' % (len(train_set), len(test_set)))
    return train_set, test_set


def gen_model_input(train_set, user_profile, seq_max_len):
    train_user_ids = np.array([line[0] for line in train_set])
    train_item_ids = np.array([line[1] for line in train_set])
    train_label = np.array([line[2] for line in train_set])
    train_seq = [line[3] for line in train_set]
    train_hist_len = np.array([line[4] for line in train_set])
    train_seq_genres = [line[5] for line in train_set]
    train_genres = np.array([line[6] for line in train_set])
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_seq_genres_pad = pad_sequences(train_seq_genres, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {
        'user_id': train_user_ids,
        'movie_id': train_item_ids,
        'hist_movie_id': train_seq_pad,
        'hist_genres': train_seq_genres_pad,
        'hist_len': train_hist_len,
        'genres': train_genres,
    }
    for key in ['gender', 'age', 'occupation', 'zip']:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values
    return train_model_input, train_label


def gen_data_set_sdm(data, seq_short_max_len=5, seq_prefer_max_len=50):
    data.sort_values('timestamp', inplace=True)
    train_set = []
    test_set = []
    for user_id, user_history in tqdm(data.groupby('user_id')):
        pos_list = user_history['movie_id'].tolist()
        genres_list = user_history['genres'].tolist()
        rating_list = user_history['rating'].tolist()
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            seq_short_len = min(i, seq_short_max_len)
            seq_prefer_len = min(max(i - seq_short_len, 0), seq_prefer_max_len)
            if i != len(pos_list) - 1:
                train_set.append(
                    (user_id, pos_list[i], 1, hist[::-1][:seq_short_len][::-1],
                     hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], seq_short_len,
                     seq_prefer_len, genres_hist[::-1][:seq_short_len][::-1],
                     genres_hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], rating_list[i]))
            else:
                test_set.append(
                    (user_id, pos_list[i], 1, hist[::-1][:seq_short_len][::-1],
                     hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], seq_short_len,
                     seq_prefer_len, genres_hist[::-1][:seq_short_len][::-1],
                     genres_hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], rating_list[i]))
    random.shuffle(train_set)
    random.shuffle(test_set)
    print('train_size=%s, test_size=%s' % (len(train_set), len(test_set)))
    return train_set, test_set


def gen_model_input_sdm(train_set, user_profile, seq_short_max_len, seq_prefer_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_iid = np.array([line[1] for line in train_set])
    train_label = np.array([line[2] for line in train_set])
    short_train_seq = [line[3] for line in train_set]
    prefer_train_seq = [line[4] for line in train_set]
    train_short_len = np.array([line[5] for line in train_set])
    train_prefer_len = np.array([line[6] for line in train_set])
    short_train_seq_genres = np.array([line[7] for line in train_set])
    prefer_train_seq_genres = np.array([line[8] for line in train_set])

    train_short_item_pad = pad_sequences(short_train_seq, maxlen=seq_short_max_len, padding='post', truncating='post',
                                         value=0)
    train_prefer_item_pad = pad_sequences(prefer_train_seq, maxlen=seq_prefer_max_len, padding='post',
                                          truncating='post',
                                          value=0)
    train_short_genres_pad = pad_sequences(short_train_seq_genres, maxlen=seq_short_max_len, padding='post',
                                           truncating='post',
                                           value=0)
    train_prefer_genres_pad = pad_sequences(prefer_train_seq_genres, maxlen=seq_prefer_max_len, padding='post',
                                            truncating='post',
                                            value=0)

    train_model_input = {'user_id': train_uid, 'movie_id': train_iid, 'short_movie_id': train_short_item_pad,
                         'prefer_movie_id': train_prefer_item_pad,
                         'prefer_sess_length': train_prefer_len,
                         'short_sess_length': train_short_len, 'short_genres': train_short_genres_pad,
                         'prefer_genres': train_prefer_genres_pad}

    for key in ['gender', 'age', 'occupation', 'zip']:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label
