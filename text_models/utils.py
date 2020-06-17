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
    path = "http://ingeotec.mx/~mgraffg/geov2/%s" % day
    try:
      request.urlretrieve(path, fname)
    except HTTPError:
      raise Exception(path)
    return fname


def download(fname, lang="Es", country=None, cache=True):
    """
    >>> from text_models.utils import download
    >>> from microtc.utils import tweet_iterator, load_model
    >>> config = list(tweet_iterator(download("config.json")))
    >>> [list(x.keys())[0] for x in config]
    ['weekday_Es', 'b4msa_Es', 'weekday_En', 'b4msa_En', 'weekday_Ar', 'b4msa_Ar']

    >>> voc = load_model(download("191225.voc", lang="Es"))
    >>> voc2 = load_model(download(config[0]["weekday_Es"]["0"][0]))

    """

    from os.path import isdir, join, isfile, dirname
    import os
    from urllib import request
    from urllib.error import HTTPError

    assert lang in ["Ar", "En", "Es"]
    diroutput = join(dirname(__file__), 'data')
    if not isdir(diroutput):
        os.mkdir(diroutput)
    if fname in ["config.json", "data.json"]:
        output = join(diroutput, fname)
        if not isfile(output) or not cache:
            request.urlretrieve("http://ingeotec.mx/~mgraffg/vocabulary/%s" % fname,
                                output)            
        return output
    if fname.count("/") == 1:
        lang, fname = fname.split("/")
    diroutput = join(diroutput, lang)
    if not isdir(diroutput):
        os.mkdir(diroutput)
    if country is not None:
        diroutput = join(diroutput, country)
        if not isdir(diroutput):
            os.mkdir(diroutput)
    output =  join(diroutput, fname)
    if not isfile(output) or not cache:
        if country is not None:
            path = "http://ingeotec.mx/~mgraffg/vocabulary/%s/%s/%s" % (lang, country, fname)
        else:
            path = "http://ingeotec.mx/~mgraffg/vocabulary/%s/%s" % (lang, fname)
        try:
            request.urlretrieve(path, output)
        except HTTPError:
            raise Exception(path)                     
    return output
    

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