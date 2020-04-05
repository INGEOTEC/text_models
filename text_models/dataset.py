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
from microtc.utils import load_model
from microtc import emoticons
from EvoMSA.utils import download
from collections import OrderedDict


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
    def __init__(self, lang="En"):
        self._lang = lang
        self._map = dict()

    @property
    def textModel(self):
        "Text model used to process the texts"

        try:
            return self._tm
        except AttributeError:
            self._tm = load_model(download("b4msa_%s.tm" % self._lang))
        return self._tm

    @staticmethod
    def load_emojis():
        def download(fname):
            from urllib import request
            import os
            output = fname.split("/")[-1]
            if os.path.isfile(output):
                return output
            request.urlretrieve(fname, output)
            return output

        data = "https://www.unicode.org/Public/emoji/12.1/emoji-data.txt"
        sec = "https://www.unicode.org/Public/emoji/12.1/emoji-sequences.txt"
        var ="https://www.unicode.org/Public/emoji/12.1/emoji-variation-sequences.txt"
        zwj = "https://www.unicode.org/Public/emoji/12.1/emoji-zwj-sequences.txt"
        emos = emoticons.read_emoji_standard(download(data))
        emoticons.read_emoji_standard(download(sec), emos)
        emoticons.read_emoji_standard(download(var), emos)
        emoticons.read_emoji_standard(download(zwj), emos)
        return {x: True for x in emos.keys()}

    def tm_words(self):
        """
        Text model words
        :rtype: dict
        """

        tm = self.textModel.text_transformations
        emos = self.load_emojis()
        words = [tm(k) for k in self.textModel.model.word2id.keys()
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
        tm = self.textModel.text_transformations
        return {tm(x): True for x in data["words"]}

    def affective_words(self):
        from ConceptModelling import text_preprocessing as base
        from microtc.utils import tweet_iterator
        import os
        
        lang = self._lang.lower()
        fname = os.path.join(os.path.dirname(base.__file__), 'data',
                             '%s.affective.words.json' % lang)
        tm = self.textModel.text_transformations
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
        text = self.textModel.text_transformations(text)
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

        text = self.textModel.text_transformations(text)
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
