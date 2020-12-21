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
from microtc.utils import load_model, tweet_iterator, Counter
from text_models.utils import download
from collections import defaultdict
from datetime import datetime
from b4msa.textmodel import TextModel
from microtc.weighting import TFIDF
from microtc.utils import SparseMatrix
from scipy.sparse import csr_matrix
from typing import List, Iterable, Union, Dict, Any, Tuple


TM_ARGS=dict(usr_option="delete", num_option="none",
             url_option="delete", emo_option="none",
             del_dup=False, del_punc=True)


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
                 tm_args=TM_ARGS):
        self._lang = lang
        self._country = country
        self._min = token_min_filter
        self._max = token_max_filter
        self._tm_args = tm_args
        self._init(data)

    def __filename(self, date):
        """
        :param date: Transform datetime instance into the filename
        :type date: str or datetime
        """
        
        if isinstance(date, datetime):
            return "%s%02i%02i.voc" % (str(date.year)[-2:], date.month, date.day)
        return date

    def __handle_day(self, day):
        """Inner function to handle the day
        
        :param day: day
        :type day: None | instance
        """

        if isinstance(day, Counter):
            return day
        if isinstance(day, dict):
            return datetime(year=day["year"],
                            month=day["month"],
                            day=day["day"])
        if hasattr(day, "year") and hasattr(day, "month") and hasattr(day, "day"):
            return datetime(year=day.year,
                            month=day.month,
                            day=day.day)
        return day         

    def _init(self, data):
        """
        Process the :py:attr:`data` to create a :py:class:`microtc.utils.Counter` 
        """

        data = self.__handle_day(data)
        if isinstance(data, str) or isinstance(data, datetime):
            self._fname = download(self.__filename(data),
                                   lang=self._lang, country=self._country)
            self.voc = load_model(self._fname)
            if isinstance(data, str):
                self._date = self.get_date(data)
            else:
                self._date = data
        elif isinstance(data, list):
            cum = self.__handle_day(data[0])
            if isinstance(cum, str) or isinstance(cum, datetime):
                cum = load_model(download(self.__filename(cum),
                                          lang=self._lang, country=self._country))
            for x in data[1:]:
                x = self.__handle_day(x)
                x = self.__filename(x)
                xx = load_model(download(x, lang=self._lang, country=self._country)) if isinstance(x, str) else x
                cum = cum + xx
            self.voc = cum
        else:
            raise Exception("%s is not handled " % type(data))
        
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

        from microtc.weighting import TFIDF
        tm = TextModel(**self._tm_args)
        tm.model = TFIDF.counter(self.voc, token_min_filter=self._min,
                                 token_max_filter=self._max)
        return tm

    def __add__(self, otro):
        _ = self.__class__([self.voc, otro.voc], lang=self._lang,
                           country=self._country, token_min_filter=self._min,
                           token_max_filter=self._max,
                           tm_args=self._tm_args)
        return _

    def __sub__(self, otro):
        voc = self.voc.copy()
        for k in otro.voc.keys():
            del voc[k]
        # - otro.voc
        _ = self.__class__([voc], lang=self._lang,
                           country=self._country, token_min_filter=self._min,
                           token_max_filter=self._max,
                           tm_args=self._tm_args)
        return _


class Tokenize(object):
    """ Tokenize transforms a text into a sequence, where 
    each number identifies a particular token; the q-grams 
    that are not found in the text are ignored.

    >>> from text_models import Tokenize
    >>> tok = Tokenize().fit(["hi~mario", "mario"])
    >>> tok.transform("good morning mario")
    [1]
    """
    def __init__(self, tm_args: Dict[str, Any]=TM_ARGS):
        self._head = dict()
        self._vocabulary = dict()
        self._tag = "__end__"
        self._textmodel = TextModel(**tm_args)

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Vocabulary used"""
        return self._vocabulary

    @property
    def textModel(self):
        """Text model, i.e., :py:class::`b4msa.text_model.TextModel`
        """

        return self._textmodel

    def fit(self, tokens: List[str]) -> 'Tokenize':
        """Train the tokenizer. 

        :param tokens: Vocabulary as a list of tokens
        :type tokens: List[str]
        """
        voc = self.vocabulary
        head = self._head
        tag = self._tag
        for word in tokens:
            if word in voc:
                continue
            current = head
            for char in word:
                try:
                    current = current[char]
                except KeyError:
                    _ = dict()
                    current[char] = _
                    current = _
            cnt = len(voc)
            voc[word] = cnt
            current[tag] = cnt
        return self

    def transform(self, texts: Union[Iterable[str], str]) -> List[Union[List[int], int]]:
        """Transform the input into a sequence where each element represents 
        a token in the vocabulary (i.e., :py:attr:`text_models.vocabulary.Tokenize.vocabulary`)"""
        func = self.textModel.text_transformations
        trans = self._transform
        if isinstance(texts, str):
            return trans(func(texts))
        return [trans(func(x)) for x in texts]

    def _transform(self, text: str) -> List[int]:
        L = []
        i = 0
        while i < len(text):
            wordid, pos = self.find(text, i=i)
            if wordid == -1:
                i += 1
                continue
            i = pos
            L.append(wordid)
        return L

    def find(self, text: str, i: int=0) -> Tuple[int, int]:
        end = i
        head = self._head
        current = head
        tag = self._tag
        wordid = -1
        while i < len(text):
            char = text[i]
            try:
                current = current[char]
                i += 1
                try:
                    wordid = current[tag]
                    end = i
                except KeyError:
                    pass
            except KeyError:
                break
        return wordid, end

    def id2word(self, id: int) -> str:
        """Token associated with id
        
        :param id: Identifier
        :type id: int
        """

        try:
            id2w = self._id2w
        except AttributeError:
            id2w = {v: k for k, v in self.vocabulary.items()}
            self._id2w = id2w
        return id2w[id]

class BagOfWords(SparseMatrix):
    """Bag of word model using TFIDF and 
    :py:class:`text_models.vocabulary.Tokenize`
    
    :param tokens: Language (Ar|En|Es) or list of tokens
    :type tokens: str|List
    """

    def __init__(self, tokens: Union[str, List[str]]="Es"):
        from microtc.utils import load_model
        from EvoMSA.utils import download
        if isinstance(tokens, list):
            xx = tokens
        else:
            xx = list(load_model(download("b4msa_%s.tm" % tokens)).model.word2id.keys())
        tok = Tokenize()
        f = lambda cdn: "~".join([x for x in cdn.split("~") if len(x)])
        tok.fit([f(k) for k in xx if k.count("~") and k[:2] != "q:"])
        tok.fit([f(k) for k in xx if k.count("~") == 0 and k[:2] != "q:"])
        qgrams = [f(k[2:]) for k in xx if k[:2] == "q:"] 
        tok.fit([x for x in qgrams if x.count("~") == 0 if len(x) >=2])
        self._tokenize = tok
        self._text = "text"

    @property
    def tokenize(self) -> Tokenize:
        """
        :py:class:`text_models.vocabulary.Tokenize` instance
        """
        
        return self._tokenize

    def get_text(self, data: Union[dict, str]) -> str:
        """Get text keywords from dict"""

        if isinstance(data, str):
            return data
        return data[self._text]

    def fit(self, X: List[Union[str, dict]]) -> 'BagOfWords':
        """ Train the Bag of words model"""
        
        from microtc.utils import Counter
        get_text = self.get_text
        cnt = Counter()
        tokens = self.tokenize.transform([get_text(x) for x in X])
        [cnt.update(x) for x in tokens]
        self._tfidf = TFIDF.counter(cnt)
        return self

    @property
    def tfidf(self)->TFIDF:
        return self._tfidf

    def id2word(self, id: int) -> str:
        """Token associated with id
        
        :param id: Identifier
        :type id: int
        """
        try:
            w_id2w = self._w_id2w
        except AttributeError:
            self._w_id2w = {v: k for k, v in self.tfidf.word2id.items()}
            w_id2w = self._w_id2w
        id = w_id2w[id]
        return self.tokenize.id2word(id)

    @property
    def num_terms(self):
        return len(self.tokenize.vocabulary)

    def _transform(self, data: List[str]) -> List[Tuple[int, float]]:
        """Transform a list of text to a Bag of Words using TFIDF"""

        data = self.tokenize.transform(data)
        tfidf = self.tfidf
        return [tfidf[x] for x in data]

    def transform(self, data: List[str]) -> csr_matrix:
        """Transform a list of text to a Bag of Words using TFIDF""" 
        return self.tonp(self._transform(data))