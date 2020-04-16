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
from text_models.vocabulary import Vocabulary


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