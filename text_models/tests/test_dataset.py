# Copyright 2019 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from numpy.testing._private.utils import assert_string_equal
from text_models.dataset import Dataset
from os.path import dirname, join
DIR = dirname(__file__)
TWEETS = join(DIR, "tweets.json.gz")

def test_dataset():
    from microtc.utils import load_model
    from EvoMSA.utils import download

    dset = Dataset()
    _ = dset.text_transformations('hola')
    assert _ == '~hola~'
    dset = Dataset(text_transformations=False)
    _ = dset.text_transformations('hola')
    assert _ == 'hola'


def test_load_emojis():
    emojis = Dataset.load_emojis()
    assert len(emojis) > 1000
    assert isinstance(emojis, dict)


def test_tm_words():
    words = Dataset().tm_words()
    assert len(words) > 1000
    assert isinstance(words, dict)


def test_aggress_words():
    words = Dataset().aggress_words()
    assert len(words)
    assert isinstance(words, dict)
    assert len([k for k in words.keys() if k.count('~')])


def test_affective_words():
    words = Dataset(lang="Es").affective_words()
    assert len(words)
    assert isinstance(words, dict)
    assert len([k for k in words.keys() if k.count('~')])


def test_add():
    dset = Dataset()
    assert len(dset.klasses) == 0
    dset.add(dset.load_emojis())
    cnt = len(dset.klasses)
    assert cnt > 0
    words = dset.tm_words()
    dset.add(words)
    print(len(dset.klasses), len(words), cnt)
    assert len(dset.klasses) <= len(words) + cnt


def test_klass():
    dset = Dataset()
    # dset.add(dset.load_emojis())
    dset.add(dset.tm_words())
    kl = dset.klasses
    xx = dset.klass("xxx good xxx morning xxT")
    for k in xx:
        assert k in kl


def test_remove():
    dset = Dataset()
    dset.add(dset.load_emojis())
    dset.add(dset.tm_words())
    xx = dset.klass("xxx good morning xxx asdfa")
    print(xx)
    assert len(xx) == 2
    dset.remove("~good~")
    xx = dset.klass("xxx good xxx morning xxx")
    print(xx)
    assert len(xx) == 1


def test_process():
    
    from microtc.emoticons import convert_emoji
    dset = Dataset()
    dset.add(dset.load_emojis())
    dset.add(dset.tm_words())
    xx = dset.process("xxx good 9 morning xxx fax x la", "~x~")
    for a, b in zip(xx, ["~xxx~good~9~morning~xxx~fax~", "~la~", "~la~"]):
        print(a, b)
        assert a == b
    txt = 'xxx good {} morning xxx fax x la'.format(convert_emoji('1F600'))
    xx = dset.process(txt, convert_emoji('1F600'))
    print(xx)
    for a, b in zip(xx, ["~xxx~good~", "~morning~xxx~fax~x~la~"]):
        assert a == b


def test_map():
    dset = Dataset()
    dset.add(dict(buenos="malos"))
    res = dset.klass("en estos buenos dias")
    print(res)
    assert "malos" in res


def test_bug_two_cons_klass():
    from EvoMSA.utils import download
    from microtc.utils import load_model
    from os.path import dirname, join
    _ = join(dirname(__file__), "..", "data", "country.ds")
    dset = load_model(_)
    r = dset.klass("mexico y usa")
    assert len(r.intersection(set(["US", "MX"]))) == 2


def test_TokenCount_process_line():
    from text_models.dataset import TokenCount
    tcount = TokenCount.bigrams()
    tcount.process_line("buenos dias xx la dias xx")
    counter = tcount.counter
    print(counter)
    assert counter["dias~xx"] == 2 and tcount.num_documents == 1


def test_TokenCount_process():
    from microtc.utils import tweet_iterator
    from text_models.dataset import TokenCount
    tcount = TokenCount.bigrams()
    tcount.process(tweet_iterator(TWEETS))
    print(tcount.counter.most_common(10))
    assert tcount.counter["in~the"] == 313


def test_TokenCount_co_occurrence():
    from microtc.utils import tweet_iterator
    from text_models.dataset import TokenCount
    tcount = TokenCount.co_ocurrence()
    tcount.process_line("buenos xxx dias")
    assert tcount.counter["dias~xxx"] == 1


def test_TokenCount_single_co_occurrence():
    from microtc.utils import tweet_iterator
    from text_models.dataset import TokenCount
    tcount = TokenCount.single_co_ocurrence()
    tcount.process_line("buenos xxx dias")
    assert tcount.counter["dias~xxx"] == 1
    assert tcount.counter["xxx"] == 1


def test_GeoFrequency():
    from text_models.dataset import GeoFrequency
    freq = GeoFrequency([])
    freq.compute_file(TWEETS)
    assert freq.data["nogeo"].counter['#earthquake~magnitude'] == 17


def test_GeoFrequency2():
    from text_models.dataset import GeoFrequency
    freq = GeoFrequency([TWEETS])
    freq.compute()
    freq.data = None
    assert freq.data is None


def test_TokenCount_clean():
    from microtc.utils import tweet_iterator
    from text_models.dataset import TokenCount
    tcount = TokenCount.single_co_ocurrence()    
    tcount.process(tweet_iterator(TWEETS))
    ant = len(tcount.counter)
    tcount.clean()
    act = len(tcount.counter)
    assert ant > act


def test_GeoFrequency_clean():
    from text_models.dataset import GeoFrequency
    freq = GeoFrequency([TWEETS])
    freq.compute()
    ant = len(freq.data)
    freq.clean()
    act = len(freq.data)
    assert ant > act


def test_Dataset_textModel_setter():
    from text_models.dataset import Dataset

    ds = Dataset(text_transformations=False)
    ds.textModel = '!'
    assert ds._tm == '!'


def test_SelfSupervisedDataset_dataset():
    from text_models.dataset import SelfSupervisedDataset
    from text_models.tests.test_dataset import TWEETS
    from EvoMSA import TextRepresentations
    
    emo = TextRepresentations(lang='es', emoji=False, dataset=False)
    semi = SelfSupervisedDataset(emo.names)
    assert len(semi.dataset.klasses) == len(emo.names)
    x = list(semi.dataset.klasses.keys())[0]
    assert x[0] == '~' and x[-1] == '~'


def test_SelfSupervisedDataset_identify_labels():
    from text_models.dataset import SelfSupervisedDataset
    from text_models.tests.test_dataset import TWEETS
    from EvoMSA import TextRepresentations
    from microtc.utils import tweet_iterator
    import os
    import gzip

    emo = TextRepresentations(lang='es', emoji=False, dataset=False)
    semi = SelfSupervisedDataset(emo.names)
    semi.identify_labels(TWEETS)
    with gzip.open(semi.tempfile, 'rb') as fpt:
        for a, b in zip(fpt.readlines(), tweet_iterator(TWEETS)):
            text, *klass = str(a, encoding='utf-8').split('|')
            assert text.strip() == emo.bow.text_transformations(b)
            break
    os.unlink(semi.tempfile)
    assert len(semi.dataset.klasses) < len(emo.names)


def test_SelfSupervisedDataset_labels_frequency():
    from text_models.dataset import SelfSupervisedDataset
    from text_models.tests.test_dataset import TWEETS
    from EvoMSA import TextRepresentations
    from microtc.utils import tweet_iterator
    import os

    emo = TextRepresentations(lang='es', emoji=False, dataset=False)
    semi = SelfSupervisedDataset(emo.names)
    semi.identify_labels(TWEETS)
    semi2 = SelfSupervisedDataset(emo.names, tempfile=semi.tempfile)
    semi2.process(TWEETS)
    assert semi2.labels_frequency is not None
    os.unlink(semi.tempfile)


def test_SelfSupervisedDataset_process():
    from text_models.dataset import SelfSupervisedDataset
    from text_models.tests.test_dataset import TWEETS
    from EvoMSA import TextRepresentations
    from microtc.utils import tweet_iterator
    from os.path import join
    import os

    emo = TextRepresentations(lang='es', emoji=False, dataset=False)
    semi = SelfSupervisedDataset(emo.names, tempfile='t.gz')
    semi.process(TWEETS)
    for k in range(len(semi.dataset.klasses)):
        os.unlink(join('', f'{k}.json'))
    os.unlink(semi.tempfile)


def test_EmojiDataset():
    from text_models.dataset import EmojiDataset
    from text_models.tests.test_dataset import TWEETS
    from os.path import join
    import os
    import gzip

    semi = EmojiDataset(tempfile='t.gz', min_num_elements=10)
    semi.process(TWEETS)
    with gzip.open('t.gz', 'rb') as fpt:
        for x in fpt:
            text = str(x, encoding='utf-8')
            assert len(text.split('|')) == 2
    for k in range(len(semi.dataset.klasses)):
        os.unlink(join('', f'{k}.json'))
    os.unlink(semi.tempfile)


def test_TrainBoW_frequency():
    from text_models.dataset import TrainBoW
    from text_models.tests.test_dataset import TWEETS
    from os.path import join
    import os
    import gzip

    bow = TrainBoW(lang='es', tempfile='t.gz')
    bow.frequency(TWEETS)
    assert len(bow.counter)
    bow = TrainBoW(lang='es', tempfile='t.gz')
    assert len(bow.counter)
    os.unlink(bow.tempfile)


def test_TrainBoW_most_common():
    from text_models.dataset import TrainBoW
    from text_models.tests.test_dataset import TWEETS
    from microtc.utils import Counter
    from os.path import join
    import os
    import gzip

    bow = TrainBoW(lang='es', tempfile='t.gz')
    bow.most_common('t2.gz', size=10, input_filename=TWEETS)
    with gzip.open('t2.gz', 'rb') as fpt:
        counter = Counter.fromjson(str(fpt.read(), encoding='utf-8'))
    assert len(counter) == 10
    os.unlink(bow.tempfile)
    os.unlink('t2.gz')


def test_TrainBoW_most_common_by_type():
    from text_models.dataset import TrainBoW
    from text_models.tests.test_dataset import TWEETS
    from microtc.utils import Counter
    from os.path import join
    import os
    import gzip

    size = 10
    bow = TrainBoW(lang='es', tempfile='t.gz')
    bow.most_common('t2.gz', size=size, input_filename=TWEETS)
    with gzip.open('t2.gz', 'rb') as fpt:
        counter = Counter.fromjson(str(fpt.read(), encoding='utf-8'))
    assert len(counter) == size
    bow = TrainBoW(lang='es', tempfile='t.gz')
    bow.most_common_by_type('t2.gz', size=size, input_filename=TWEETS)
    with gzip.open('t2.gz', 'rb') as fpt:
        counter2 = Counter.fromjson(str(fpt.read(), encoding='utf-8'))
    inter = set(counter.keys()).intersection(counter2.keys())
    assert len(inter) < size
    os.unlink(bow.tempfile)
    os.unlink('t2.gz')