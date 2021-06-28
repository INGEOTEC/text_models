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
    from datetime import datetime
    if isinstance(day, dict):
        return datetime(year=day["year"],
                        month=day["month"],
                        day=day["day"])
    if hasattr(day, "year") and hasattr(day, "month") and hasattr(day, "day"):
        return datetime(year=day.year,
                        month=day.month,
                        day=day.day)
    raise Exception("Not implemented: %s" % day)


def download_tokens(day, lang:str= "Es", country: str=None, cache: bool=True) -> str:
    from os.path import isdir, join, isfile, dirname
    import os
    from urllib import request
    from urllib.error import HTTPError
    day = handle_day(day)
    date = "%s%02i%02i" % (day.year, day.month, day.day)
    date_str = "%s/%02i/%02i" % (str(day.year)[2:], day.month, day.day) 
    diroutput = join(dirname(__file__), 'data')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    diroutput = join(diroutput, lang)
    if not isdir(diroutput):
        os.mkdir(diroutput)        
    if country is None:
        fname = join(diroutput, date + ".gz")
        if isfile(fname) and cache:
            return fname 
        path = "https://github.com/INGEOTEC/tokens-data-%s/raw/main/%s/%s/nogeo.gz" % (day.year, lang, date_str)
    else:
        fname = join(diroutput, country + "_" + date + ".gz")
        if isfile(fname) and cache:
            return fname 
        path = "https://github.com/INGEOTEC/tokens-data-%s/raw/main/%s/%s/%s.gz" % (day.year, lang, date_str, country)
    try:
      request.urlretrieve(path, fname)
    except HTTPError:
      raise Exception(path)
    return fname        


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
        self.words_N = sum([v for k, v in voc.items() if k.count("~") == 0])
        self.bigrams_N = sum([v for k, v in voc.items() if k.count("~")])

    def compute(self, bigram: str) -> float:
        a, b = bigram.split("~")
        a = self.voc.get(a, 1) / self.words_N
        b = self.voc.get(b, 1) / self.words_N
        bar_x = self.voc[bigram] / self.bigrams_N
        num = bar_x - (a * b)
        den = np.sqrt(bar_x / self.bigrams_N)
        return num / den