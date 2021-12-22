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
from collections import defaultdict
from microtc.utils import load_model, Counter
from b4msa.textmodel import TextModel
from microtc.weighting import TFIDF
from microtc.utils import SparseMatrix
from scipy.sparse import csr_matrix
from typing import List, Iterable, OrderedDict, Union, Dict, Any, Tuple
from text_models.utils import download_tokens, handle_day


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
    :param states: Whether to keep the state or accumulate the information on the country
    :type states: bool

    >>> from text_models.vocabulary import Vocabulary
    >>> day = dict(year=2020, month=2, day=14)
    >>> voc = Vocabulary(day, lang="En", country="US")
    """

    def __init__(self, data, lang: str="Es", 
                 country: str=None, states: bool=False) -> None:
        self._lang = lang
        self._country = country
        self._states = states
        if isinstance(data, dict) and len(data) > 3:
            self._data = data
        else:
            self.date = data
            self._init(data)
        if not states:
            self._n_words = sum([v for k, v in self.voc.items() if k.count("~") == 0])
            self._n_bigrams = sum([v for k, v in self.voc.items() if k.count("~")])

    def probability(self):
        """Transform frequency to a probability"""
        voc = self.voc
        for k in voc:
            num = voc[k]
            if k.count("~"):
                den = self._n_bigrams
            else:
                den = self._n_words
            voc[k] = num / den

    def _init(self, data):
        """
        Process the :py:attr:`data` to create a :py:class:`microtc.utils.Counter` 
        """

        def sum_vocs(vocs):
            voc = vocs[0]
            for v in vocs[1:]:
                voc = voc + v
            return voc

        if isinstance(data, list):
            vocs = [download_tokens(day, lang=self._lang, country=self._country)
                    for day in data]
            vocs = [load_model(x) for x in vocs]
            if isinstance(vocs[0], Counter):
                voc = sum_vocs(vocs)
            elif not self._states:
                vocs = [sum_vocs([v for _, v in i]) for i in vocs]
                voc = sum_vocs(vocs)
            else:
                voc = {k: v for k, v in vocs[0]}
                for v in vocs[1:]:
                    for k, d in v:
                        try:
                            voc[k] = voc[k] + d
                        except KeyError:
                            voc[k] = d
            self._data = voc
        else:
            self.voc = load_model(download_tokens(data, lang=self._lang, country=self._country))

    @property
    def date(self):
        """
        Date obtained from the filename, on multiple files, this is not available.
        """

        return self._date

    @date.setter
    def date(self, d):
        if isinstance(d, list):
            self._date = None
            return
        self._date = handle_day(d)

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
        if not isinstance(d, list):
            self._data = d
            return
        if self._states:
            self._data = {k: v for k, v in d}
            return
        aggr = d[0][1]
        for _, v in d[1:]:
            aggr = aggr + v
        self._data = aggr
        
    def common_words(self, quantile: float=None, bigrams=True):
        """Words used frequently; these correspond to py:attr:`EvoMSA.base.EvoMSA(B4MSA=True)`
        In the case quantile is given the these words and bigrams correspond to 
        the most frequent.
        """

        if quantile is None:
            from EvoMSA.utils import download
            return load_model(download("b4msa_%s.tm" % self._lang)).model.word2id
        words_N = sum([v for k, v in self.voc.items() if k.count("~") == 0])
        score = [[k, v / words_N] for k, v in self.voc.items() if k.count("~") == 0]
        score.sort(key=lambda x: x[1], reverse=True)
        cum, k = 0, 0
        while cum <= quantile:
            cum += score[k][1]
            k += 1
        output = [k for k, _ in score[:k]]
        if bigrams:
            bigrams_N = sum([v for k, v in self.voc.items() if k.count("~")])
            score_bi = [[k, v / bigrams_N] for k, v in self.voc.items() if k.count("~")]
            score_bi.sort(key=lambda x: x[1], reverse=True)
            cum, k = 0, 0
            while cum <= quantile:
                cum += score_bi[k][1]
                k += 1
            output += [k for k, _ in score_bi[:k]]
        return output

    @staticmethod
    def _co_occurrence(word: str, voc: dict) -> dict:
        D = dict()
        for k, v in voc.items():    
            if k.count("~") == 0:
                continue
            a, b = k.split("~")
            if a != word and b != word:
                continue
            key = a if a != word else b
            D[key] = v
        return D

    def co_occurrence(self, word: str) -> dict:
        if self._states:
            return {k: self._co_occurrence(word, v) for k, v in self.voc.items()}
        return self._co_occurrence(word, self.voc)

    def day_words(self) -> "Vocabulary":
        """Words used on the same day of different years"""
        
        from datetime import date, datetime

        hoy = date.today()
        hoy = datetime(year=hoy.year, month=hoy.month, day=hoy.month)
        L = []
        for year in range(2015, hoy.year + 1):
            try:
                curr = datetime(year=year, month=self.date.month, day=self.date.day)
            except ValueError:
                continue
            if (curr - self.date).days == 0:
                continue
            try:
                download_tokens(curr, lang=self._lang, country=self._country)
            except Exception:
                continue
            L.append(curr)
        if len(L) == 0:
            return None
        return self.__class__(L if len(L) > 1 else L[0],
                              lang=self._lang,
                              country=self._country,
                              states=self._states)

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
        _ = self.__class__(r, lang=self._lang,
                           country=self._country,
                           states=self._states)

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

    def remove(self, words: dict, bigrams=True) -> None:
        """
        Remove the words from the current vocabulary
        
        :param words: Tokens
        """
        if not bigrams:
            voc = self.voc
            for w in words:
                try:
                    del voc[w]
                except Exception:
                    continue
            return
        K = []
        for k in self.voc:
            if k.count("~"):
                a, b = k.split("~")
                if a in words or b in words:
                    K.append(k)
            if k in words:
                K.append(k)
        for k in K:
            del self.voc[k]  

    def histogram(self, min_elements: int=30, words: bool=False):
        group = defaultdict(list)
        [group[v].append(k) for k, v in self.voc.items() if words or k.count("~")]
        keys = list(group.keys())
        keys.sort()
        lst = list()
        hist = OrderedDict()
        for k in keys:
            _ = group[k]
            if len(lst) + len(_) >= min_elements:
                hist[k] = lst + _
                lst = list()
                continue
            lst += _
        if len(lst):
            hist[k] = lst
        return hist

    @staticmethod
    def available_dates(dates=List, n=int, countries=List, lang=str):
        """Retrieve the first n dates available for all the countries

        :param dates: List of dates
        :param n: Number of days
        :param countries: List of countries
        :lang lang: Language
        """

        missing = Counter(countries) if countries is not None else None
        rest = []
        dates = dates[::-1]
        while len(dates) and (len(rest) < n or n == -1):
          day = dates.pop()
          flag = True
          iter = missing.most_common() if missing is not None else [[None, None]]
          for country, _ in iter:
            try:
                download_tokens(day, lang=lang, country=country)
            except Exception:
              flag = False
              if missing is not None:
                  missing.update([country])  
              break
          if flag:
            rest.append(day)
        return rest


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

    @textModel.setter
    def textModel(self, v):
        self._textmodel = v

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
    >>> from EvoMSA.tests.test_base import TWEETS
    >>> from microtc.utils import tweet_iterator
    >>> from text_models.vocabulary import BagOfWords
    >>> tw = list(tweet_iterator(TWEETS))
    >>> BoW = BagOfWords().fit(tw)
    >>> BoW['hola mundo']
    [(758, 0.7193757438600711), (887, 0.6946211479258095)]
    """

    def __init__(self, tokens: Union[str, List[str]]="Es"):
        from microtc.utils import load_model
        from EvoMSA.utils import download
        tok = Tokenize()
        if isinstance(tokens, list):
            xx = tokens
        else:
            textModel = load_model(download("b4msa_%s.tm" % tokens))
            xx = list(textModel.model.word2id.keys())
            tok.textModel = textModel
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

    def __getitem__(self, data: str):
        _ = self.tokenize.transform(data)
        return self.tfidf[_]
