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
from scipy.stats import pearsonr
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
        request.urlretrieve(path, output)          
    return output
    

        
