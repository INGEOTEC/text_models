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
from text_models.utils import download
from text_models.vocabulary import Vocabulary, Tokenize, BagOfWords


def test_init():
    from datetime import datetime
    voc = Vocabulary("191225.voc", lang="En")
    assert voc.date.day == 25
    voc2 = Vocabulary(["191225.voc", "191226.voc"], lang="En")
    assert len(voc2.voc) > len(voc.voc)
    voc = Vocabulary(datetime(year=2018, month=12, day=24), lang="Es")
    assert voc.date.day == 24 and voc.date.year == 2018



def test_create_text_model():
    voc = Vocabulary("191225.voc")
    nterms = len(voc.voc)
    tm = voc.create_text_model()
    assert len(tm["buenos dias"]) > 4
    assert nterms == tm.num_terms
    # assert nterms == len(voc.voc)


def test_weekday_words():
    voc = Vocabulary("191225.voc")
    words = voc.weekday_words()
    assert len(words) > 10000


def test_common_words():
    voc = Vocabulary("191225.voc")
    words = voc.common_words()
    assert len(words) > 10000


def test_remove():
    voc = Vocabulary("191225.voc")
    numterms = len(voc.voc)
    voc.remove(voc.weekday_words())
    assert numterms > len(voc.voc)
    numterms = len(voc.voc)
    voc.remove(voc.common_words())
    assert numterms > len(voc.voc)


def test_day_words():
    voc = Vocabulary("200229.voc", lang="En")
    words = voc.day_words()
    assert len(words) > 10000


def test_remove_qgrams():
    voc = Vocabulary("191225.voc")
    voc.remove_qgrams()
    assert len([x for x in voc.voc if x[:2] == "q:"]) == 0


def test_previous_day():
    from os.path import basename

    voc = Vocabulary("200301.voc")
    prev = voc.previous_day()
    assert basename(prev._fname) == "200229.voc"


def test_dict_functions():
    voc = Vocabulary("200301.voc")

    assert len(voc) == len([x for x in voc])
    data = [k for k, v in voc.items()]
    assert len(voc) == len(data)
    assert data[0] in voc
    assert "BLABLA" not in voc
    assert voc.get("BLABLA") == 0
    assert voc[data[0]] == voc.get(data[0])


def test_remove_emojis():
    voc = Vocabulary("200301.voc")
    voc.remove_qgrams()
    voc.remove_emojis()


def test_country():
    date = "151219.voc"
    voc = Vocabulary(date, lang="Es", country="MX")
    assert len(voc)
    voc2 = Vocabulary(date, lang="Es")
    print(len(voc2), len(voc))
    assert voc2.voc.update_calls > voc.voc.update_calls


def test_vocabulary_dict():
    class D(object):
        def __init__(self, year, month, day):
            self.year = year
            self.month = month
            self.day = day

    voc = Vocabulary(dict(year=2020, month=7, day=21))
    assert voc["buenos"]
    voc2 = Vocabulary(D(2020, 7, 21))
    assert voc["buenos"] == voc2["buenos"]


def test_vocabulary_data_lst():
    import pandas as pd

    d = list(pd.date_range("2020-07-20", "2020-07-21"))
    print(len(d))
    vocs = Vocabulary(d)
    assert vocs["buenos"]
    assert len(d) == 2


def test_add():
    voc1 = Vocabulary(dict(year=2020, month=7, day=21), token_min_filter=0)
    voc2 = Vocabulary(dict(year=2020, month=7, day=1), token_min_filter=0)
    r = voc1 + voc2
    r2 = Vocabulary([dict(year=2020, month=7, day=21),
                     dict(year=2020, month=7, day=1)])
    print(len(r.voc), len(voc1.voc), len(voc2.voc))
    print(len(r.voc), len(r2.voc))
    assert len(r.voc) > len(voc1.voc) and len(r.voc) > len(voc2.voc)


def test_sub():
    voc1 = Vocabulary(dict(year=2020, month=7, day=21), token_min_filter=0)
    voc2 = Vocabulary(dict(year=2020, month=7, day=1), token_min_filter=0)
    r = voc1 - voc2
    r2 = Vocabulary([dict(year=2020, month=7, day=21),
                     dict(year=2020, month=7, day=1)])
    print(len(r.voc), len(voc1.voc), len(voc2.voc))
    print(len(r.voc), len(r2.voc))
    assert len(r.voc) < len(voc1.voc) and len(r.voc) < len(voc2.voc)    


def test_tokenize_fit():
    # voc1 = Vocabulary(dict(year=2020, month=7, day=21), token_min_filter=0)
    # print(voc1.voc)
    tok = Tokenize().fit(["hola", "holas", "mma", "ma~xa", "hola"])
    assert len(tok.vocabulary) == 4
    assert tok.vocabulary["ma~xa"] == 3


def test_tokenize_find():
    # voc1 = Vocabulary(dict(year=2020, month=7, day=21), token_min_filter=0)
    # print(voc1.voc)
    tok = Tokenize().fit(["hola", "holas", "mma", "ma~xa", "hola"])
    cdn = "holas~mmado~hola"
    wordid, pos = tok.find(cdn)
    assert wordid == 1 and pos == 5
    wordid, pos = tok.find(cdn, i=pos)
    assert wordid == -1
    wordid, pos = tok.find(cdn, i=pos+1)
    assert wordid == 2
    wordid, pos = tok.find("xa")
    assert wordid == -1


def test_tokenize__transform():
    # voc1 = Vocabulary(dict(year=2020, month=7, day=21), token_min_filter=0)
    # print(voc1.voc)
    tok = Tokenize().fit(["hola", "holas", "mma", "ma~xa", "hola"])
    cdn = "holas~mmado~hola~xa~ma~xa~hola"
    _ = tok._transform(cdn)
    for a, b in zip(_, [1, 2, 0, 3, 0]):
        assert a == b


def test_tokenize_transform():
    # voc1 = Vocabulary(dict(year=2020, month=7, day=21), token_min_filter=0)
    # print(voc1.voc)
    tok = Tokenize().fit(["hola", "holas", "mma", "ma~xa", "hola"])
    cdn = "holas mmado hola xa ma xa hola"
    _ = tok.transform(cdn)
    for a, b in zip(_, [1, 2, 0, 3, 0]):
        assert a == b
    _ = tok.transform([cdn])
    assert len(_) == 1
    assert len(_[0]) == 5


def test_BagOfWords_init():
    from microtc.utils import load_model
    from EvoMSA.utils import download    
    tm = BagOfWords()
    xx = list(load_model(download("b4msa_Es.tm")).model.word2id.keys())
    tm2 = BagOfWords(tokens=xx)
    assert len(tm.tokenize.vocabulary) == len(tm2.tokenize.vocabulary)
    # inv = {v: k for k, v in tm.tokenize.vocabulary.items()}
    # print(len(inv))
    # print([inv[x] for x in tm.tokenize.transform("buenos dias mujer madres zorra")])
    # print([x for x in tm.tokenize.vocabulary.keys() if len(x) == 1])
    # assert False


def test_BagOfWords_fit():
    from EvoMSA.tests.test_base import TWEETS
    from microtc.utils import tweet_iterator
    from scipy.sparse import csr_matrix

    X = list(tweet_iterator(TWEETS))
    bg = BagOfWords().fit(X)
    bg = BagOfWords().fit([x["text"] for x in X])
    xx = bg._transform(["buenos y felices dias"])
    print(len(xx), len(xx[0]), xx)
    assert len(xx) == 1 and len(xx[0]) == 3 and len(xx[0][1]) == 2
    xx = bg.transform(["buenos y felices dias"])
    assert isinstance(xx, csr_matrix)
    # inv = bg.id2word
    # print([(inv(k), v) for k, v in bg.tfidf[xx[0]]])
    # assert False
    # print(bg._cnt)
    # 
    # for k, v in bg._cnt.most_common(10):
    #     print(inv[k], v)
    # assert False