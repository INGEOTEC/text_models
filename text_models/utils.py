# Copyright 2020 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.metrics import f1_score, recall_score
from scipy.stats import pearsonr, norm
import numpy as np
from datetime import datetime, timedelta
from typing import List
from microtc.params import OPTION_DELETE, OPTION_NONE
from microtc.utils import load_model
from os.path import isdir, join, isfile, dirname
import os
from urllib import request
from urllib.error import HTTPError

MICROTC='2.4.2'
TM_ARGS=dict(usr_option=OPTION_DELETE, num_option=OPTION_NONE,
             url_option=OPTION_DELETE, emo_option=OPTION_NONE,
             hashtag_option=OPTION_NONE,
             del_dup=False, del_punc=True)


def get_text(tw: dict):
    """Get the text from a twitter
    >>> from text_models.utils import get_text
    >>> from text_models.tests.test_dataset import TWEETS
    >>> from microtc.utils import tweet_iterator
    >>> D = list(tweet_iterator(TWEETS))
    >>> get_text(D[2])
    '@TheNitron @TechAltar Ooh I see still a cool car have you pushed it too 200 on the autobahn yet if so how does the acceleration feel when doing so?'
    >>> get_text(D[1])
    '@melk150 Kkkkkkk bom diaaa'
    """
    if tw.get('truncated', False):
        # text = tw.get('extended_tweet', dict())
        text = tw['extended_tweet']
        # return text.get('full_text', '')
        return text['full_text']
    return tw.get('text', '')


def macro_f1(y, hy):
    """
    >>> from text_models.utils import macro_f1
    >>> y = [0, 1, 1, 0]
    >>> hy = [0, 1, 0, 0]
    >>> macro_f1(y, hy)
    0.7333333333333334
    """

    return f1_score(y, hy, average="macro")


def macro_recall(y, hy):
    """
    >>> from text_models.utils import macro_recall
    >>> y = [0, 1, 1, 0]
    >>> hy = [0, 1, 0, 0]
    >>> macro_recall(y, hy)
    0.75
    """

    return recall_score(y, hy, average="macro")


def pearson(y, hy):
    """
    >>> from text_models.utils import pearson
    >>> y = [0, 1, 1, 0]
    >>> hy = [0, 1, 0, 0]
    >>> pearson(y, hy)
    0.5773502691896258
    """

    r = pearsonr(y, hy)[0]
    if np.isfinite(r):
        return r
    return 0


def download_geo(day):
    """
    >>> from text_models.utils import download_geo
    >>> fname = download_geo("200101.travel")
    """
    from os.path import isdir, join, isfile, dirname
    import os
    from urllib import request
    from urllib.error import HTTPError

    diroutput = dirname(__file__)
    for d in ["data", "geo"]:
        diroutput = join(diroutput, d)
        if not isdir(diroutput):
            os.mkdir(diroutput)
    fname = join(diroutput, day)
    if isfile(fname):
      return fname
    path = "https://github.com/INGEOTEC/mobility-data/raw/main/data/%s" % day
    try:
      request.urlretrieve(path, fname)
    except HTTPError:
      raise Exception(path)
    return fname


def handle_day(day):
    if isinstance(day, dict):
        return datetime(year=day["year"],
                        month=day["month"],
                        day=day["day"])
    if hasattr(day, "year") and hasattr(day, "month") and hasattr(day, "day"):
        return datetime(year=day.year,
                        month=day.month,
                        day=day.day)
    raise Exception("Not implemented: %s" % day)


def date_range(init=dict, end=dict) -> List[datetime]:
    """Interval of day
    :param init: Start of the sequence
    :type init: dict with year, month, and day as keywords
    :param end: End of the sequence (exclusive)
    :type init: dict with year, month, and day as keywords    
    """

    init = handle_day(init)
    end = handle_day(end)
    dates = []
    while init <= end:
      dates.append(init)
      init = init + timedelta(days=1)        
    return dates


def download_tokens(day, lang:str= "Es", country: str='nogeo',
                    cache: bool=True) -> str:
    assert country is not None
    assert lang is not None
    country = country.upper() if country != 'nogeo' else country.lower()
    lang = lang.lower()
    day = handle_day(day)
    date = "%s%02i%02i" % (day.year, day.month, day.day)
    diroutput = join(dirname(__file__), 'data')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    diroutput = join(diroutput, 'releases')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    diroutput = join(diroutput, date)
    if not isdir(diroutput):
        os.mkdir(diroutput)
    data_fname = '{}-{}.gz'.format(lang, country)
    fname = join(diroutput, data_fname)
    if isfile(fname) and cache:
        return fname 
    path = 'https://github.com/INGEOTEC/text_models/releases/download/{}/{}'.format(date, data_fname)
    try:
      request.urlretrieve(path, fname)
    except HTTPError:
      raise Exception(path)
    return fname


def load_bow(lang='es'):
    """
    Download and load the Bag of Word text representation

    :param lang: ['ar', 'zh', 'en', 'fr', 'pt', 'ru', 'es']
    :type lang: str
    >>> from text_models.utils import load_bow
    >>> bow = load_bow(lang='en')
    >>> repr = bow['hi']
    """
    lang = lang.lower().strip()
    assert lang in ['ar', 'zh', 'en', 'fr', 'pt', 'ru', 'es']
    diroutput = join(dirname(__file__), 'models')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    fname = join(diroutput, f'{lang}_{MICROTC}.microtc')
    if not isfile(fname):
        path = f'https://github.com/INGEOTEC/text_models/releases/download/models/{lang}_{MICROTC}.microtc'
        try:
            request.urlretrieve(path, fname)
        except HTTPError:
            raise Exception(path)    
    return load_model(fname)


def _load_text_repr(lang='es', name='emo', k=0):
    diroutput = join(dirname(__file__), 'models')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    fname = join(diroutput, f'{lang}_{name}_{k}_muTC{MICROTC}.LinearSVC')
    if not isfile(fname):
        path = f'https://github.com/INGEOTEC/text_models/releases/download/models/{lang}_{name}_{k}_muTC{MICROTC}.LinearSVC'
        try:
            request.urlretrieve(path, fname)
        except HTTPError:
            raise Exception(path)    
    return load_model(fname)


def load_emoji(lang='es', emoji=0):
    """
    Download and load the Emoji representation

    :param lang: ['ar', 'zh', 'en', 'fr', 'pt', 'ru', 'es']
    :type lang: str
    :param emoji: emoji identifier
    :type emoji: int

    >>> from text_models.utils import load_emoji, load_bow
    >>> bow = load_bow(lang='en')
    >>> emo = load_emoji(lang='en', emoji=0)
    >>> X = bow.transform(['this is funny'])
    >>> df = emo.decision_function(X)
    """
    lang = lang.lower().strip()
    assert lang in ['ar', 'zh', 'en', 'fr', 'pt', 'ru', 'es']
    return _load_text_repr(lang, 'emo', emoji)


def emoji_information(lang='es'):
    """
    Download and load the Emoji statistics

    :param lang: ['ar', 'zh', 'en', 'fr', 'pt', 'ru', 'es']
    :type lang: str

    >>> from text_models.utils import emoji_information
    >>> info = emoji_information()
    >>> info['💧']
    {'recall': 0.10575916230366492, 'ratio': 0.0003977123419509893, 'number': 3905}
    """
    lang = lang.lower().strip()
    assert lang in ['ar', 'zh', 'en', 'fr', 'pt', 'ru', 'es']
    diroutput = join(dirname(__file__), 'models')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    data = []
    for ext in ['info', 'perf']:
        fname = join(diroutput, f'{lang}_emo.{ext}')
        if not isfile(fname):
            path = f'https://github.com/INGEOTEC/text_models/releases/download/models/{lang}_emo.{ext}'
            try:
                request.urlretrieve(path, fname)
            except HTTPError:
                raise Exception(path)    
        data.append(load_model(fname))
    uno, dos = data
    [v.update(dict(number=uno[k])) for k, v in dos.items()]
    return dos
    

def load_dataset(lang='es', name='HA', k=0):
    """
    Download and load the Emoji representation

    :param lang: ['ar', 'zh', 'en', 'es']
    :type lang: str
    :param emoji: emoji identifier
    :type emoji: int

    >>> from text_models.utils import load_dataset, load_bow
    >>> bow = load_bow(lang='en')
    >>> ds = load_dataset(lang='en', name='travel')
    >>> X = bow.transform(['this is funny'])
    >>> df = ds.decision_function(X)
    """
    lang = lang.lower().strip()
    assert lang in ['ar', 'zh', 'en', 'es']
    return _load_text_repr(lang, name, k)    


def dataset_information(lang='es'):
    """
    Download and load datasets information

    :param lang: ['ar', 'zh', 'en', 'es']
    :type lang: str

    >>> from text_models.utils import emoji_information
    >>> info = dataset_information()
    """
    lang = lang.lower().strip()
    assert lang in ['ar', 'zh', 'en', 'es']
    diroutput = join(dirname(__file__), 'models')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    data = []
    ext = 'info'
    fname = join(diroutput, f'{lang}_dataset.{ext}')
    if not isfile(fname):
        path = f'https://github.com/INGEOTEC/text_models/releases/download/models/{lang}_dataset.{ext}'
        try:
            request.urlretrieve(path, fname)
        except HTTPError:
            raise Exception(path)    
    data = load_model(fname)
    return data


class Gaussian(object):
    """
    Gaussian distribution
    
    >>> from text_models.utils import Gaussian
    >>> g = Gaussian().fit([1, 2, 1.4, 2.1, 2.3])
    >>> g.predict_proba([2.15])
    array([0.42050953])

    """
    def fit(self, X):
        """
        Fit the model 
        
        :param X: Data
        :type X: list

        """

        self._mu = np.mean(X)
        self._std = np.std(X)
        return self

    def predict_proba(self, X):
        """
        Predict the probability of X

        :param X: Data
        :type X: list
        """

        X = np.atleast_1d(X)
        r = norm.cdf(X, loc=self._mu, scale=self._std)
        m = r > 0.5
        r[m] = 1 - r[m]
        return 2 * r


def remove_outliers(data):
    """Remove outliers using boxplot algorithm

    >>> from text_models.utils import remove_outliers
    >>> _ = remove_outliers([0.2, 3.0, 2, 2.5, 5.4, 3.2])
    """
    import numpy as np
    data = np.atleast_1d(data)
    q1 = np.quantile(data, q=0.25)
    q3 = np.quantile(data, q=0.75)
    iqr = q3 - q1
    upper = q3 + iqr * 1.5 
    lower = q1 - iqr * 1.5
    m = (data > lower) & (data < upper)
    return data[m]


class MobilityException(Exception):
    pass


class MobilityTransform(object):
    """
    Transform travel data
    """

    def __init__(self, data=None):
        self._data = data
        self._travel_ins = None

    @property
    def mobility_instance(self):
        """Mobility instance"""

        return self._travel_ins

    @property
    def data(self):
        """Data to be used on the transformation"""
        return self._data

    @data.setter
    def data(self, value):
        _ = {k: np.median(v) for k, v in value.items()}
        if sum([1 for v in _.values() if v != 0]) < len(value):
            raise MobilityException("%s" % len(value))
        self._data = _

    @mobility_instance.setter
    def mobility_instance(self, data):
        self._travel_ins = data
        self._wdays = np.array([d.weekday() for d in data.dates])

    def transform(self, data):
        wdays = self._wdays
        data = np.atleast_1d(data)
        r = np.zeros(data.shape)
        for wd in range(7):
            m = wdays == wd
            _ = (data[m] - self.data[wd]) / self.data[wd]
            r[m] = _
        return r * 100.


class TStatistic(object):
    def __init__(self, voc: dict) -> None:
        self.voc = voc
        freq = dict()
        for x, v in voc.items():
            try:
                a, b = x.split("~")
            except ValueError:
                continue
            freq[a] = freq.get(a, 0) + v
            freq[b] = freq.get(b, 0) + v
        self.freq = freq
        self.N = sum(freq.values())
        self.Nbigrams = sum([v for k, v in voc.items() if k.count("~")])

    def compute(self, bigram: str) -> float:
        a, b = bigram.split("~")
        a = self.freq.get(a, 1) / self.N
        b = self.freq.get(b, 1) / self.N
        bar_x = self.voc[bigram] / self.Nbigrams
        num = bar_x - (a * b)
        den = np.sqrt(bar_x * (1 - bar_x) / self.Nbigrams)
        return num / den


class LikelihoodRatios(TStatistic):
    def __init__(self, voc: dict, independent: bool=True) -> None:
        super().__init__(voc)
        self.independent = independent 

    def compute(self, bigram: str) -> float:
        def L(k, n, x):
            _ = k * np.log(x) + (n - k) * np.log(1 - x)
            return _
        a, b = bigram.split("~")
        c1 = self.freq.get(a, 1)
        c2 = self.freq.get(b, 1)
        c12 = self.voc[bigram]
        N = self.N
        p = c2 / N
        if self.independent:
            bar_x = c12 / self.Nbigrams
            if p * c1 / self.N > bar_x:
                return 0
        p1 = c12 / c1
        p2 = (c2 - c12) / (N - c1)
        if p1 >= 1:
            p1 = 0.999
        if p2 >= 1:
            p2 = 0.999
        if p2 <= 0:
            p2 = 1e-5
        t1 = L(c12, c1, p) + L(c2 - c12, N - c1, p)
        t2 = L(c12, c1, p1) + L(c2 - c12, N - c1, p2)
        return -2 * (t1 - t2)
