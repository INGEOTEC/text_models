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

from text_models.utils import TM_ARGS
from text_models.dataset import Dataset
import microtc
from microtc import TextModel
from microtc.utils import load_model, save_model
from glob import glob
from os.path import join
from random import shuffle
import numpy as np
from tqdm import tqdm
from text_models.inhouse import data
from text_models.inhouse.data import num_tweets_language
from os.path import dirname, basename
from collections import Counter
from sklearn.svm import LinearSVC

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
               join('models', f'{lang}_{microtc.__version__}.microtc'))
    return tm


def count_emo(lang='zh'):
    fnames = glob(join('data', lang, 'emo', '*.gz'))
    cnt = Counter()
    for fname in fnames:
        for key, data in load_model(fname).items():
            [cnt.update(x['klass'])
             for x in data if len(x['klass']) == 1]
    return cnt


def emo(k, lang='zh', size=2**19):
    ds = Dataset(text_transformations=False)
    ds.add(ds.load_emojis())    
    output = join('models', f'{lang}_emo_{k}_mu{microtc.__version__}')
    dd = load_model(join('models', f'{lang}_emo.info'))
    _ = [x for x, v in dd.most_common() if v >= 2**10]
    tot = sum([v for x, v in dd.most_common() if v >= 2**10])
    if k >= len(_):
        return
    pos = _[k]
    neg = set([x for i, x in enumerate(_) if i != k])
    POS, NEG, ADD = [], [], []
    for fname in glob(join('data', lang, 'emo', '*.gz')):
        for key, data in load_model(fname).items():
            for d in data:
                klass = d['klass']
                if len(klass) == 1:
                    klass = klass.pop()
                    if klass == pos:
                        POS.append(ds.process(d['text']))
                    elif klass in neg:
                        NEG.append(ds.process(d['text']))
                elif tot < size:
                    if pos not in klass and len(klass.intersection(neg)):
                        ADD.append(ds.process(d['text']))
    shuffle(POS), shuffle(NEG), shuffle(ADD)
    size2 = size // 2
    POS = POS[:size2]
    if len(NEG) < size2:
        NEG.extend(ADD)
    NEG = NEG[:size2]
    y = [1] * len(POS)
    y.extend([-1] * len(NEG))
    tm = load_model(join('models', f'{lang}_{microtc.__version__}.microtc'))
    X = tm.transform(POS + NEG)
    m = LinearSVC().fit(X, y)
    save_model(m, f'{output}.LinearSVC')


# if __name__ == '__main__':
#     cnt = count_emo(lang='es')
# if __name__ == '__main__':
#     tm = bow(lang='zh')
