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
from microtc.utils import load_model, tweet_iterator
from text_models.utils import download
from collections import defaultdict


class Vocabulary(object):
    """
    Vocabulary class is used to transform the tokens and their 
    respective frequencies in a Text Model, as well as, to analyze 
    the tokens obtained from tweets collected.

    This class can be used to replicate some of the Text Models
    developed for :py:class:`EvoMSA.base.EvoMSA`.

    :param data: Tokens and their frequencies  
    :type data: str or list
    :param lang: Language (Ar, En, or Es)
    :type lang: str
    :param country: Two letter country code
    :type country: str
    :param token_min_filter: Minimum frequency
    :type token_min_filter: float | int
    :param token_max_filter: Maximum frequency
    :type token_max_filter: float | int
    :param tm_args: Text-model parameters
    :type tm_args: dict

    >>> from text_models.vocabulary import Vocabulary
    >>> voc = Vocabulary("191225.voc", lang="En")
    """

    def __init__(self, data, lang="Es", country=None,
                 token_min_filter=0.001,
                 token_max_filter=0.999,
                 tm_args=dict(usr_option="delete", num_option="none",
                              url_option="delete", emo_option="none",
                              del_dup=False, del_punc=True)):
        self._lang = lang
        self._country = country
        self._min = token_min_filter
        self._max = token_max_filter
        self._tm_args = tm_args
        self._init(data)

    def _init(self, data):
        """
        Process the :py:attr:`data` to create a :py:class:`microtc.utils.Counter` 
        """

        if isinstance(data, str):
            self._fname = download(data, lang=self._lang, country=self._country)
            self.voc = load_model(self._fname)
            self._date = self.get_date(data)
        elif isinstance(data, list):
            cum = data.pop()
            if isinstance(cum, str):
                cum = load_model(download(cum, lang=self._lang, country=self._country))
            for x in data:
                xx = load_model(download(x, lang=self._lang, country=self._country)) if isinstance(x, str) else x
                cum = cum + xx
            self.voc = cum
        
    @staticmethod
    def get_date(filename):
        """
        Obtain the date from the filename. The format is YYMMDD.

        :param filename: Filename
        :type filename: str
        :rtype: datetime
        """
        import datetime

        d = filename.split("/")[-1].split(".")[0]
        return datetime.datetime(year=int(d[:2]) + 2000,
                                 month=int(d[2:4]),
                                 day=int(d[-2:]))

    @property
    def date(self):
        """
        Date obtained from the filename, on multiple files, this is not available.
        """

        return self._date

    @property
    def weekday(self):
        """
        Weekday
        """

        return str(self.date.weekday())

    @property
    def voc(self):
        """Vocabulary, i.e., tokens and their frequencies"""

        return self._data

    @voc.setter
    def voc(self, d):
        from microtc.weighting import TFIDF
        TFIDF.filter(d, token_min_filter=self._min, token_max_filter=self._max)
        self._data = d

    def common_words(self):
        """Words used frequently; these correspond to py:attr:`EvoMSA.base.EvoMSA(B4MSA=True)`"""

        from EvoMSA.utils import download
        return load_model(download("b4msa_%s.tm" % self._lang)).model.word2id
       
    def weekday_words(self):
        """Words group by weekday"""

        from EvoMSA.utils import download
        return load_model(download("weekday-%s_%s.voc" % (self.weekday, self._lang)))

    def day_words(self):
        """Words used on the same day of different years"""
        
        import datetime

        dd = list(tweet_iterator(download("data.json", cache=False)))[0][self._lang]
        day = defaultdict(list)
        [day[x[2:6]].append(x) for x in dd]
        date = self.date
        dd = day["%02i%02i" % (date.month, date.day)]
        curr = "%s%02i%02i.voc" % (str(date.year)[:2],
                                   date.month, date.day)
        dd = [x for x in dd if x != curr]
        if len(dd) == 0:
            one_day = datetime.timedelta(days=1)
            r = date - one_day
            dd = day["%02i%02i" % (r.month, r.day)]
        dd = [download(x, lang=self._lang, country=self._country) for x in dd]
        return self.__class__(dd, token_min_filter=self._min,
                              token_max_filter=self._max)

    def __iter__(self):
        for x in self.voc:
            yield x

    def remove_emojis(self):
        """Remove emojis"""
        from .dataset import Dataset
        data = Dataset()
        data.add(data.load_emojis())
        keys = [(k, [x for x in data.klass(k) if not x.isnumeric()])  for k in self]
        keys = [(k, v) for k, v in keys if len(v) and v[0] != "#"]
        for k, v in keys:
            del self.voc[k]

    def previous_day(self):
        """Previous day"""

        import datetime

        one_day = datetime.timedelta(days=1)
        r = self.date - one_day
        fname = "%s%02i%02i.voc" % (str(r.year)[-2:], r.month, r.day)
        _ = self.__class__(fname, lang=self._lang, country=self._country,
                           token_min_filter=self._min,
                           token_max_filter=self._max)
        return _

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, key):
        return self.voc[key]

    def __contains__(self, key):
        return key in self.voc

    def get(self, data, defaultvalue=0):
        """Frequency of data"""

        return self.voc.get(data, defaultvalue)

    def items(self):
        """Items of :py:attr:`self.voc`"""

        return self.voc.items()

    def remove(self, words):
        """
        Remove the words from the current vocabulary
        
        :param words: Tokens
        """

        for x in words:
            if x in self.voc:
                del self.voc[x]

    def remove_qgrams(self):
        """Remove the q-grams in the vocabulary"""

        keys = [k for k in self.voc if k[:2] == "q:"]
        for k in keys:
            del self.voc[k]

    def create_text_model(self):
        """
        Create a text model using :py:class:`b4msa.textmodel.TextModel`

        >>> from text_models.utils import download
        >>> from microtc.utils import tweet_iterator
        >>> from text_models.vocabulary import Vocabulary
        >>> conf = tweet_iterator(download("config.json", cache=False))
        >>> conf = [x for x in conf if "b4msa_En" in x][0]
        >>> # Files to create b4msa_En.tm text model
        >>> data = conf["b4msa_En"]
        >>> # Taking only a few to reduce the time
        >>> data = data[:10]
        >>> voc = Vocabulary(data, lang="En")
        >>> tm = voc.create_text_model()
        """

        from b4msa.textmodel import TextModel
        from microtc.weighting import TFIDF
        tm = TextModel(**self._tm_args)
        tm.model = TFIDF.counter(self.voc, token_min_filter=self._min,
                                 token_max_filter=self._max)
        return tm

