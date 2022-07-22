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

from text_models.inhouse.reader import TweetIterator
from text_models.dataset import GeoFrequency, Dataset
from microtc.utils import tweet_iterator, save_model, load_model
import os
from os.path import join, dirname, basename, isdir, isfile
from glob import glob
from microtc import TextModel
from text_models.utils import TM_ARGS
from collections import defaultdict
from joblib import delayed, Parallel
import random

JSON = join(dirname(__file__), '..', '..', 'data', '*.json')


def num_tweets_language(lang='es'):
    output = []
    for fname in glob(JSON):
        data = list(tweet_iterator(fname))[0].get(lang, None)
        if data is None:
            continue
        tot = sum(data.values())
        fname = basename(fname)
        date = dict(year=int(fname[:4]), month=int(fname[4:6]),
                    day=int(fname[6:8]))
        output.append([date, tot])
    return output


def choose(days: list, size=100e6):
    random.shuffle(days)
    cnt = 0
    output = list()
    while len(days) and cnt < size:
        ele = days.pop()
        if ele[1] <= 1:
            continue
        cnt += ele[1]
        output.append(ele)
    return output


class Process(object):
    def __init__(self) -> None:
        self.tm = TextModel(**TM_ARGS)
        self.data = []

    def process_line(self, tweet):
        text = tweet['text']
        d = self.tm.text_transformations(text)
        if len(d) > 3:
            _ = dict(text=text, id=tweet['id'])
            self.data.append(_)


def create_output_path(lang):
    output = 'data'
    if not isdir(output):
        os.mkdir(output)
    output = join(output, lang)
    if not isdir(output):
        os.mkdir(output)
    return output


def store_tweets(lang, date, output_path=create_output_path):
    output = output_path(lang)
    _ = '{year:d}{month:02d}{day:02d}.gz'.format(**date)
    output = join(output, _)
    if isfile(output):
        return
    tw_iterator = TweetIterator(lang)
    freq = GeoFrequency([], reader=tw_iterator.tweet_iterator)
    freq.data = defaultdict(Process)
    freq.compute_file(date)
    output_dict = defaultdict(list)
    for k, v in freq.data.items():
        key = k.split('-')[0]
        output_dict[key].extend(v.data) 
    save_model(output_dict, output)
    return output


def emo_data(lang='zh'):
    fnames = glob(join('data', lang, '*.gz'))
    ds = Dataset(text_transformations=False)
    ds.add(ds.load_emojis())
    for fname in fnames:
        output = dict()
        for key, tweets in load_model(fname).items():
            labels = [ds.klass(x['text']) for x in tweets]
            inner = []
            for tweet, label in zip(tweets, labels):
                if len(label) == 0:
                    continue
                tweet['klass'] = label
                inner.append(tweet)
            if len(inner):
                output[key] = inner
        if len(output) == 0:
            continue
        output_fname = join(dirname(fname), 'emo')
        if not isdir(output_fname):
            os.mkdir(output_fname)
        output_fname = join(output_fname, basename(fname))
        save_model(output, output_fname)


def create_test(lang='zh', n_jobs=16):
    def output_path(lang):
        output = join('data', lang, 'test')
        if not isdir(output):
            os.mkdir(output)
        return output

    _ = join('data', lang, '{year:d}{month:02d}{day:02d}.gz')
    func = lambda x: isfile(_.format(**x))
    data = [[d, n] for d, n in num_tweets_language(lang=lang)
            if not func(d) and n > 0]
    days = choose(data, size=10e6)
    output_path(lang)
    Parallel(n_jobs=n_jobs)(delayed(store_tweets)(lang, day, output_path=output_path)
                        for day, _ in days)

# if __name__ == '__main__':
#     data = emo_data(lang='zh')

# if __name__ == '__main__':
#     from joblib import delayed, Parallel
#     LANG = 'zh'
#     data = num_tweets_language(lang=LANG)
#     days = choose(data)
#     create_output_path(LANG)
#     fnames = Parallel(n_jobs=32)(delayed(store_tweets)(LANG, day) for day, _ in days)