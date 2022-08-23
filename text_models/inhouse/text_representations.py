# Copyright 2022 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from text_models.utils import TM_ARGS, MICROTC, load_bow
from text_models.dataset import Dataset
from microtc import TextModel
from microtc.utils import load_model, save_model, tweet_iterator
from glob import glob
from os.path import join
from random import shuffle
import numpy as np
from tqdm import tqdm
from text_models.inhouse import data
from text_models.inhouse.data import num_tweets_language
from os.path import dirname, basename, isfile
from collections import Counter
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
from sklearn.metrics import recall_score

data.JSON = join(dirname(__file__), '..', '..', 'data', '*.json')


def data_bow(lang='zh', size=2**19):
    num_tweets = {'{year}{month:02d}{day:02d}.gz'.format(**k): v 
                  for k, v in num_tweets_language(lang=lang)}
    files = [[num_tweets[basename(x)], x] for x in glob(join('data', lang, '*.gz'))]
    files.sort(key=lambda x: x[0])
    files = [x[1] for x in files]
    per_file = size / len(files)
    output = []
    for k, file in tqdm(enumerate(files), total=len(files)):
        tweets = load_model(file)
        [shuffle(tweets[key]) for key in tweets]
        cnt = [[key, len(tweets[key])] for key in tweets]
        cnt.sort(key=lambda x: x[1])
        per_place = int(np.ceil(per_file // len(cnt)))
        prev = len(output)
        for i, (key, n) in enumerate(cnt):
            _ = [x['text'] for x in tweets[key][:per_place]]
            output.extend(_)
            if len(_) < per_place and i < len(cnt) - 1:
                per_place += int(np.ceil((per_place - len(_)) / (len(cnt) - (i + 1))))
        inc = len(output) - prev
        if inc < per_file and k < len(files) - 1:
            per_file += (per_file - inc) / (len(files) - (k + 1))
    shuffle(output)        
    return output


def bow(lang='zh', num_terms=2**14):
    tweets = data_bow(lang=lang)
    token_min_filter = 0
    if lang == 'zh':
        token_list = [1, 2, 3]
        q_grams_words = False
        # token_min_filter=0.0005
    else:
        token_list = [-1, 2, 3, 4]
        q_grams_words = True
        # token_min_filter=0.001
    tm = TextModel(token_list=token_list,
                   token_max_filter=len(tweets),
                   token_min_filter=token_min_filter, 
                   q_grams_words=q_grams_words,
                   **TM_ARGS).fit(tweets)

    model = tm.model
    id2word = {v: k for k, v in model.word2id.items()}
    N = model._ndocs
    word_weight = [[N / 2**v , id2word[k]]
                for k, v in model.wordWeight.items()]
    word_weight.sort(key=lambda x: x[0], reverse=True)
    word_weight = word_weight[:num_terms]
    model.word2id = {token: k for k, (w, token) in enumerate(word_weight)}
    model.wordWeight = {k: w for k, (w, token) in enumerate(word_weight)}

    save_model(tm,
               join('models', f'{lang}_{MICROTC}.microtc'))
    return tm


def count_emo(lang='zh'):
    fnames = glob(join('data', lang, 'emo', '*.gz'))
    cnt = Counter()
    for fname in fnames:
        for key, data in load_model(fname).items():
            [cnt.update(x['klass'])
             for x in data if len(x['klass']) == 1]
    return cnt


def emo(k, lang='zh', size=2**19, n_jobs=8):
    def read_data(fname):
        P = []
        N = []
        for key, data in load_model(fname).items():
            for d in data:
                klass = d['klass']
                if len(klass) == 1:
                    klass = klass.pop()
                    if klass == pos:
                        P.append(ds.process(d['text']))
                    elif klass in neg:
                        N.append(ds.process(d['text']))
        shuffle(N)
        return P, N[:len(P)]

    ds = Dataset(text_transformations=False)
    ds.add(ds.load_emojis())    
    output = join('models', f'{lang}_emo_{k}_muTC{MICROTC}')
    dd = load_model(join('models', f'{lang}_emo.info'))
    _ = [x for x, v in dd.most_common() if v >= 2**10]
    tot = sum([v for x, v in dd.most_common() if v >= 2**10])
    if k >= len(_):
        return
    pos = _[k]
    neg = set([x for i, x in enumerate(_) if i != k])
    POS, NEG = [], []
    _ = Parallel(n_jobs=n_jobs)(delayed(read_data)(fname) 
                                for fname in tqdm(glob(join('data',
                                                            lang,
                                                            'emo',
                                                            '*.gz'))))
    for P, N in _:
        POS.extend(P)
        NEG.extend(N)
    size2 = size // 2
    if size2 > len(POS):
        size2 = len(POS)
    if size2 > len(NEG):
        size2 = len(NEG)
    # assert len(NEG) >= len(POS)
    shuffle(POS), shuffle(NEG)    

    POS = POS[:size2]
    NEG = NEG[:len(POS)]
    y = [1] * len(POS) + [-1] * len(NEG)
    # tm = load_model(join('models', f'{lang}_{MICROTC}.microtc'))
    tm = load_bow(lang=lang)
    X = tm.transform(POS + NEG)
    m = LinearSVC().fit(X, y)
    save_model(m, f'{output}.LinearSVC')


def recall_emo(lang='zh', n_jobs=1):
    def predict(fname, ds, tm, emoji):
        D = []
        for key, tweets in load_model(fname).items():
            labels = [ds.klass(x['text']) for x in tweets]
            _ = [[x['text'], label] for label, x in zip(labels, tweets)
                    if len(klasses.intersection(label))]
            D.extend(_)
        X = tm.transform([x for x, _ in D])
        y = [y for _, y in D]
        hy = []
        for k, emo in enumerate(emoji):
            output = join('models', f'{lang}_emo_{k}_muTC{MICROTC}')
            m = load_model(f'{output}.LinearSVC')
            hy.append(m.predict(X))
        return y, hy

    def performance(emo, y, hy):
        y_emo = [emo in i for i in y]
        perf = recall_score(y_emo, hy > 0, pos_label=True)
        return perf, sum(y_emo) / len(y)
        
    info = load_model(join('models', f'{lang}_emo.info'))
    info = [[k, v] for k, (v, _) in enumerate(info.most_common()) if _ >= 2**10]
    klasses = set([v for k, v in info])
    fnames = glob(join('data', lang, 'test', '*.gz'))
    ds = Dataset(text_transformations=False)
    ds.add(ds.load_emojis())
    dd = load_model(join('models', f'{lang}_emo.info'))
    emoji = [x for x, v in dd.most_common() if v >= 2**10]    
    # tm = load_model(join('models', f'{lang}_{MICROTC}.microtc'))
    tm = load_bow(lang=lang)
    predictions = Parallel(n_jobs=n_jobs)(delayed(predict)(fname, ds, tm, emoji)
                                          for fname in fnames)
    y = []
    [y.extend(x) for x, hy in predictions]
    hys = np.vstack([np.vstack(hy).T for _, hy in predictions])
    output = dict()
    _ = Parallel(n_jobs=n_jobs)(delayed(performance)(emo, y, hy)
                                for emo, hy in zip(emoji, hys.T))
    output = {emo: {'recall': perf, 'ratio': ratio} 
              for emo, (perf, ratio) in zip(emoji, _)}
    save_model(output, join('models', f'{lang}_emo.perf'))


def dataset(lang, fname, name):
    D = list(tweet_iterator(fname))
    labels = np.unique([x['klass'] for x in D])
    if isfile(join('models', f'{lang}_{name}_0_muTC{MICROTC}.LinearSVC')):
        return labels
    tm = load_bow(lang=lang)
    X = tm.transform(D)
    for k, label in enumerate(labels):
        output = join('models', f'{lang}_{name}_{k}_muTC{MICROTC}')    
        pos_index = np.array([i for i, x in enumerate(D) if x['klass'] == label])
        neg_index = np.array([i for i, x in enumerate(D) if x['klass'] != label])
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)
        nele = min(pos_index.shape[0], neg_index.shape[0])
        pos_index = pos_index[:nele]
        neg_index = neg_index[:nele]
        index = np.concatenate((pos_index, neg_index))
        y = [1] * nele + [-1] * nele
        m = LinearSVC().fit(X[index], y)
        save_model(m, f'{output}.LinearSVC')
    return labels

# if __name__ == '__main__':
#     cnt = count_emo(lang='es')
# if __name__ == '__main__':
#     tm = bow(lang='zh')
