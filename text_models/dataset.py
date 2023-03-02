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
from b4msa.textmodel import TextModel
from microtc.utils import load_model
from microtc import emoticons
from microtc.utils import Counter
from microtc.params import OPTION_DELETE, OPTION_NONE
from microtc.utils import tweet_iterator
from EvoMSA.utils import download, b4msa_params
from EvoMSA import BoW
from text_models.place import BoundingBox, location
from text_models.utils import get_text, TM_ARGS
from sklearn.svm import LinearSVC
from collections import OrderedDict, defaultdict
from joblib import Parallel, delayed
from typing import List, Iterable, Callable, Union, Dict
from os.path import join, dirname, isfile
import numpy as np
import gzip
import tempfile
import json


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


class Dataset(object):
    """
    Self-supervised learning requires the automatic construction of a 
    labeled dataset. This class contains different methods to facilitate 
    the build of this type of dataset, starting from an unlabeled corpus. 

    >>> from text_models.dataset import Dataset
    >>> dataset = Dataset()
    >>> dataset.add(dataset.load_emojis())
    >>> dataset.add(dataset.tm_words())
    >>> result = dataset.klass("good morning Mexico")
    >>> dataset.process("good morning Mexico")
    ['~', '~mexico~']
    """
    def __init__(self, lang="En", text_transformations=True):
        self._lang = lang
        self._map = dict()
        self._text_transformations = text_transformations

    @property
    def textModel(self):
        "Text model used to process the texts"

        try:
            return self._tm
        except AttributeError:
            self._tm = TextModel(**TM_ARGS)
        return self._tm

    @textModel.setter
    def textModel(self, value):
        self._tm = value

    @property
    def text_transformations(self):
        try:
            flag = self._text_transformations
        except AttributeError:
            flag = True
        if flag:
            return self.textModel.text_transformations
        return lambda x: x

    @staticmethod
    def load_emojis():
        fname = join(dirname(__file__), 'data', 'emojis.dict')
        return load_model(fname)

    def tm_words(self):
        """
        Text model words
        :rtype: dict
        """

        tm = self.text_transformations
        emos = self.load_emojis()
        textModel = load_model(download("b4msa_%s.tm" % self._lang))
        words = [tm(k) for k in textModel.model.word2id.keys()
                 if k[:2] != "q:" and k.count("~") == 0 and k not in emos]
        words.sort()
        _ = OrderedDict([(w, True) for w in words])
        return _

    def aggress_words(self):
        from EvoMSA import base
        from microtc.utils import tweet_iterator
        import os
        
        lang = self._lang.lower()
        fname = os.path.join(os.path.dirname(base.__file__), 'conf',
                             'aggressiveness.%s' % lang)
        data = list(tweet_iterator(fname))[0]
        tm = self.text_transformations
        return {tm(x): True for x in data["words"]}

    def affective_words(self):
        from EvoMSA.ConceptModelling import text_preprocessing as base
        from microtc.utils import tweet_iterator
        import os
        
        lang = self._lang.lower()
        fname = os.path.join(os.path.dirname(base.__file__), 'data',
                             '%s.affective.words.json' % lang)
        tm = self.text_transformations
        words = dict()
        for data in tweet_iterator(fname):
            words.update({tm(x): True for x in data["words"]})
        return words

    @property
    def klasses(self):
        """Labels or words"""

        try:
            return self._words
        except AttributeError:
            self._words = OrderedDict()
        return self._words

    def add(self, data):
        """
        Add words to the processor

        :param data: words
        :type data: dict
        """
        
        self._map.update({k: v for k, v in data.items() if not isinstance(v, bool)})
        words = self.klasses
        words.update(data)
        if hasattr(self, "_data_structure"):
            del self._data_structure

    @property
    def data_structure(self):
        try:
            return self._data_structure
        except AttributeError:
            _ = emoticons.create_data_structure
            self._data_structure = _(self.klasses)
        return self._data_structure

    def klass(self, text):
        """
        Labels in a text

        :param text:
        :type text: str
        :returns: The labels in the text
        :rtype: set
        """

        get = self._map.get
        text = self.text_transformations(text)
        lst = self.find_klass(text)
        _ = [text[a:b] for a, b in lst]
        return set([get(x, x) for x in _])

    def find_klass(self, text):
        """Obtain the position of each label in the text

        :param text: text
        :type text: str
        :return: list of pairs, init and end of the word
        :rtype: list
        """

        blocks = list()
        init = i = end = 0
        head = self.data_structure
        current = head
        text_length = len(text)
        while i < text_length:
            char = text[i]
            try:
                current = current[char]
                i += 1
                if "__end__" in current:
                    end = i
            except KeyError:
                current = head
                if end > init:
                    blocks.append([init, end])
                    if (end - init) > 2 and text[end - 1] == '~':
                        init = i = end = (end - 1)
                    else:
                        init = i = end
                elif i > init:
                    if (i - init) > 2 and text[i - 1] == '~':
                        init = end = i = (i - 1)
                    else:                    
                        init = end = i
                else:
                    init += 1
                    i = end = init
        if end > init:
            blocks.append([init, end])
        return blocks

    def process(self, text, klass=None):
        """
        Remove klass from text

        :param text:
        :type text: str
        :param klass:
        :type klass: str
        :rtype: list
        """

        text = self.text_transformations(text)
        lst = self.find_klass(text)
        if klass is not None:
            lst = [[a, b] for a, b in lst if text[a:b] == klass]
        lst.reverse()
        init = 0
        B = []
        text_len = len(text)
        while len(lst):
            a, b = lst.pop()
            if (b - a) > 2:
                if a < text_len and text[a] == "~" and a > 0:
                    a += 1
                if b > 0 and text[b-1] == "~" and b < text_len:
                    b -= 1
            B.append(text[init:a])
            init = b
        if init < len(text):
            B.append(text[init:])
        return [x for x in B if len(x)]

    def remove(self, klass):
        """
        Remove label from the processor

        :param klass:
        :type klass: str
        """

        del self.klasses[klass]
        if hasattr(self, "_data_structure"):
            del self._data_structure


class TokenCount(object):
    """Count frequency"""

    def __init__(self, tokenizer: Callable[[Union[str, dict]], Iterable[str]]) -> None:
        self._tokenizer = tokenizer
        self._counter = Counter()

    @property
    def counter(self) -> Counter:
        return self._counter

    @property
    def num_documents(self) -> int:
        return self.counter.update_calls

    def process(self, iterable: Iterable[Union[str, dict]]) -> None:
        pl = self.process_line
        [pl(line) for line in iterable]

    def process_line(self, txt: Union[str, dict]) -> None:
        self.counter.update(self._tokenizer(txt))

    def clean(self) -> None:
        counter = self.counter
        min_value = 0.0001 * counter.update_calls
        min_value = max(2, min_value)
        keys = list(counter.keys())
        for k in keys:
            if counter[k] <= min_value:
                del counter[k]        

    @staticmethod
    def textModel(token_list) -> TextModel:
        kwargs = TM_ARGS.copy()
        kwargs["usr_option"] = OPTION_NONE
        kwargs["num_option"] = OPTION_DELETE
        tm = TextModel(token_list=token_list, **kwargs)
        return tm

    @classmethod
    def bigrams(cls) -> "TokenCount":
        tm = cls.textModel(token_list=[-2])
        return cls(tokenizer=tm.tokenize)

    @classmethod
    def co_ocurrence(cls) -> "TokenCount":
        tm = cls.textModel(token_list=[-1])
        def co_ocurrence(txt):
            tokens = tm.tokenize(txt)
            for k, frst in enumerate(tokens[:-1]):
                for scnd in tokens[k+1:]:
                    if frst == scnd:
                        yield frst
                    else:
                        _ = [frst, scnd]
                        _.sort()
                        yield "~".join(_)
        return cls(tokenizer=co_ocurrence)

    @classmethod
    def single_co_ocurrence(cls) -> "TokenCount":
        tm = cls.textModel(token_list=[-1])
        def co_ocurrence(txt):
            tokens = tm.tokenize(txt)
            for k, frst in enumerate(tokens[:-1]):
                for scnd in tokens[k+1:]:
                    if frst != scnd:
                        _ = [frst, scnd]
                        _.sort()
                        yield "~".join(_)
            for x in tokens:
                yield x
        return cls(tokenizer=co_ocurrence)


class Place(object):
    """
    >>> tweet = {'place': {'country_code': 'MX', 'bounding_box': {'type': 'Polygon', 'coordinates': [[[-99.067766, 19.366862], [-99.067766, 19.498582], [-98.966267, 19.498582], [-98.966267, 19.366862]]]}}}
    >>> place = Place()
    >>> place(tweet)
    'MX-MEX'
    """
    def __init__(self) -> None:
        self._label = BoundingBox().label
        _ = join(dirname(__file__), "data", "state.dict")
        self._states = load_model(_)

    def __call__(self, tweet: dict) -> Union[str, None]:
        label = self._label
        states = self._states
        country = None
        try:
            place = tweet.get('place', dict())
            if place is None:
                return None
            country = place['country_code']
        except KeyError:
            return None
        if country is None or not len(country):
            return None
        try:
            geo = label(dict(position=location(tweet), country=country))
        except Exception:
            return country
        try:
            geo = states[geo]
        except KeyError:
            return country
        return geo
        

class GeoFrequency(object):
    def __init__(self, fnames: Union[list, str],
                       reader: Callable[[str], Iterable[dict]]=tweet_iterator) -> None:
        self._fnames = fnames if isinstance(fnames, list) else [fnames]
        self._reader = reader
        self._place = Place()
        # self._label = BoundingBox().label
        self._data = defaultdict(TokenCount.single_co_ocurrence)
        # _ = join(dirname(__file__), "data", "state.dict")
        # self._states = load_model(_)

    @property
    def data(self) -> defaultdict:
        return self._data

    @data.setter
    def data(self, value: defaultdict) -> None:
        self._data = value

    def compute(self) -> None:
        for fname in tqdm(self._fnames):
            self.compute_file(fname)

    def compute_file(self, fname: str) -> None:
        # label = self._label
        # states = self._states
        place = self._place
        data = self.data
        for line in self._reader(fname):
            geo = place(line)
            key = geo if geo is not None else 'nogeo'
            data[key].process_line(line)
            # try:
            #     country, geo = None, None
            #     country = line["place"]["country_code"]
            #     geo = label(dict(position=location(line), country=country))
            #     geo = states[geo]
            # except Exception:
            #     pass
            # if geo is not None:
            #     data[geo].process_line(line)
            # elif country is not None:
            #     data[country].process_line(line)
            # else:
            #     data["nogeo"].process_line(line)

    def clean(self) -> None:
        keys = list(self.data.keys())
        data = self.data
        _ = [x.num_documents for x in data.values()]
        max_value = max(_) if len(_) else 0
        min_value = 0.0001 * max_value
        min_value = max(2, min_value)
        for key in keys:
            data[key].clean()
            if len(data[key].counter) == 0 or data[key].num_documents <= min_value:
                del data[key]

class TrainBoW(object):
    def __init__(self, lang:str, tempfile: str=tempfile.mktemp()) -> None:
        self._tempfile = tempfile        
        self._lang = lang

    @property
    def lang(self):
        return self._lang

    @property
    def tempfile(self):
        return self._tempfile
    
    @property
    def counter(self):
        try:
            return self._counter
        except AttributeError:
            if isfile(self.tempfile):
                with gzip.open(self.tempfile, 'rb') as fpt:
                    counter = Counter.fromjson(str(fpt.read(), encoding='utf-8'))
            else:
                counter = Counter()
        self._counter = counter
        return self._counter
    
    def frequency(self, filename: str, size=2**20):
        if isfile(self.tempfile) and len(self.counter):
            return
        tm = TextModel(**b4msa_params(lang=self.lang))
        counter = self.counter
        for i, tweet in zip(tqdm(range(size)),
                            tweet_iterator(filename)):
            counter.update(set(tm.tokenize(tweet)))
        with gzip.open(self.tempfile, 'wb') as fpt:
            fpt.write(bytes(counter.tojson(), encoding='utf-8'))

    def delete_freq_N(self, counter):
        borrar = []
        for k, v in counter.most_common():
            if v < counter.update_calls:
                break
            borrar.append(k)
        for x in borrar:
            del counter[x]

    def most_common(self, output_filename: str, 
                    size: int, input_filename: str=None):
        self.frequency(input_filename)
        counter = self.counter
        self.delete_freq_N(counter)
        borrar = []
        [borrar.append(k) for k, _ in counter.most_common()[size:]]
        for x in borrar:
            del counter[x]
        with gzip.open(output_filename, 'wb') as fpt:
            fpt.write(bytes(counter.tojson(), encoding='utf-8'))

    def most_common_by_type(self, output_filename: str, 
                            size: int, input_filename: str=None):
        def key(k):
            if k[:2] == 'q:':
                return f'q:{len(k) - 2}'
            elif '~' in k:
                return 'bigrams'
            return 'words'        
        
        self.frequency(input_filename)
        counter = self.counter
        self.delete_freq_N(counter)

        tot = dict()
        for k, v in counter.items():
            _key = key(k)
            tot[_key] = tot.get(_key, 0) + v
        tot = {k: np.log2(v) for k, v in tot.items()}
        norm = [(k, np.log2(v) - tot[key(k)]) 
                for k, v in counter.most_common()]
        norm.sort(key=lambda x: x[1], reverse=True)
        for k, _ in norm[size:]:
            del counter[k]

        with gzip.open(output_filename, 'wb') as fpt:
            fpt.write(bytes(counter.tojson(), encoding='utf-8'))



class SelfSupervisedDataset(object):
    """Create a masked language model
    >>> from text_models.dataset import Dataset, SelfSupervisedDataset
    >>> from text_models.tests.test_dataset import TWEETS
    >>> from EvoMSA import TextRepresentations
    >>> emo = TextRepresentations(lang='es', emoji=False, dataset=False)
    >>> semi = SelfSupervisedDataset(emo.names)
    >>> semi.process(TWEETS)
    """
    def __init__(self, labels: List[str],
                 dataset_class: Dataset=Dataset,
                 dataset_kwargs: dict=dict(text_transformations=False),
                 bow: BoW=BoW(lang='es'),    
                 words: bool=True,
                 reader: Iterable=tweet_iterator,               
                 num_elements: int=2**17,
                 min_num_elements: int=2**10,
                 tempfile: str=tempfile.mktemp(),
                 n_jobs: int=1) -> None:
        self.labels = labels
        self._dataset_class = dataset_class
        self._dataset_kwargs = dataset_kwargs
        self._bow = bow
        self._words = words
        self._reader = reader
        self._num_elements = num_elements
        self._min_num_elements = min_num_elements
        self._tempfile = tempfile
        self._n_jobs = n_jobs
        self.add_labels(labels)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    def dataset_instance(self):
        return self._dataset_class(**self._dataset_kwargs)

    @property
    def dataset(self):
        try:
            return self._dataset
        except AttributeError:
            self._dataset = self.dataset_instance()
        return self._dataset

    @property
    def bow(self):
        return self._bow

    @property
    def reader(self):
        return self._reader

    @property
    def words(self):
        return self._words

    @words.setter
    def words(self, value):
        self._words = value
    
    @property 
    def num_elements(self):
        return self._num_elements

    @property
    def min_num_elements(self):
        return self._min_num_elements

    @property
    def tempfile(self):
        return self._tempfile

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def labels_frequency(self):
        try:
            return self._labels_frequency
        except AttributeError:
            return None

    @labels_frequency.setter
    def labels_frequency(self, value):
        self._labels_frequency = value
        D = []
        for k, v in self.dataset.klasses.items():
            if value[v] < self.min_num_elements:
                D.append(k)
        [self.dataset.remove(x) for x in D]

    def add_labels(self, labels: Iterable[str]):
        self.bow.bow.disable_text_transformations = False
        tt = self.bow.bow.text_transformations if self.words else lambda x: x
        labels = [tt(x) for x in self.labels]
        self.dataset.add({k: v for k, v in zip(labels, self.labels)})

    def select_labels_for_text(self, labels, labels_freq):
        return True

    def identify_labels(self, filename: str, cache_size: int=1024):
        def flush(D):
            while D:
                text, labels = D.pop(0)
                fpt.write(bytes(f'{text}|{labels}\n', encoding='utf-8'))
            fpt.flush()

        klass = self.dataset.klass
        self.bow.bow.disable_text_transformations = False
        tt = self.bow.bow.text_transformations
        counter = Counter()
        size = self.num_elements
        inv = {v: k for k, v in self.dataset.klasses.items()}        
        with gzip.open(self.tempfile, 'wb') as fpt:
            D = []
            for tweet in self.reader(filename):
                text = tt(tweet)
                labels = klass(text)
                if len(labels) == 0:
                    continue
                if not self.select_labels_for_text(labels, counter):
                    continue
                counter.update(labels)
                D.append((text, '|'.join(labels)))
                for k, v in counter.items():
                    if v >= size and inv[k] in self.dataset.klasses:
                        self.dataset.remove(inv[k])
                if len(D) == cache_size:
                    flush(D)
            if len(D):
                flush(D)
        keys = list(self.dataset.klasses.keys())
        [self.dataset.remove(x) for x in keys]
        self.add_labels(self.labels)
        self.labels_frequency = counter

    def test_positive(self, label, labels, neg):
        if label in labels:
            return 1
        if len(labels - neg) < len(labels):
            return -1
        return 0

    def process_label(self, k, output):
        _ = list(self.dataset.klasses.items())
        key, label = _.pop(k)
        neg = set([v for k, v in _])
        ds = self.dataset_instance()
        ds.add({key: label})
        ds.textModel = self.bow.bow
        size = self.num_elements
        POS, NEG = [], []
        with gzip.open(self.tempfile, 'rb') as fpt:
            for a in fpt:
                text, *klass = str(a, encoding='utf-8').strip().split('|')
                flag = self.test_positive(label, set(klass), neg)
                if flag == 1 and len(POS) < size:
                    POS.append(ds.process(text))
                elif flag == -1 and len(NEG) < size:
                    NEG.append(text)
                if len(POS) == size and len(NEG) == size:
                    break
        _min = min(len(POS), len(NEG))
        POS = POS[:_min]
        NEG = NEG[:_min]
        self.train_classifier(POS, NEG, k, output, label)

    def train_classifier(self, POS, NEG, k, output, label):     
        y = [1] * len(POS) + [-1] * len(NEG)
        X = self.bow.bow.transform(POS + NEG)
        m = LinearSVC().fit(X, y)
        with open(join(output, f'{k}.json'), 'w') as fpt:
            coef = m.coef_[0].tolist()
            intercept = m.intercept_[0]
            _ = json.dumps(dict(coef=coef, intercept=intercept, labels=[-1, label]))
            print(_, file=fpt)

    def count_labels_frequency(self):
        counter = Counter()
        with gzip.open(self.tempfile, 'rb') as fpt:
            for a in fpt:
                text, *klass = str(a, encoding='utf-8').strip().split('|')
                counter.update(klass)
        self.labels_frequency = counter

    def process(self, filename: str=None, identify_labels_kwargs: dict=dict(),
                output: str=''):
        if not isfile(self.tempfile):
            self.identify_labels(filename, **identify_labels_kwargs)
        if self.labels_frequency is None:
            self.count_labels_frequency()
        self.bow.bow.disable_text_transformations = True
        Parallel(n_jobs=self.n_jobs)(delayed(self.process_label)(k, output=output)
                                     for k in tqdm(range(len(self.dataset.klasses))))
        self.bow.bow.disable_text_transformations = False


class EmojiDataset(SelfSupervisedDataset):
    def __init__(self, labels: List[str]=[], 
                       words: bool = False, 
                       **kwargs) -> None:
        assert len(labels) == 0
        super(EmojiDataset, self).__init__(labels, words=words, **kwargs)
        emojis = self.dataset.load_emojis()
        self.labels = emojis
        self.add_labels(emojis)

    def add_labels(self, labels: Dict):
        if len(labels):
            self.dataset.add(labels)

    def select_labels_for_text(self, labels, labels_freq):
        return len(labels) == 1
