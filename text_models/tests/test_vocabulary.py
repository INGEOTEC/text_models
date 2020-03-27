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
    voc = Vocabulary("191225.voc", lang="En")
    assert voc.date.day == 25
    voc2 = Vocabulary(["191225.voc", "191226.voc"], lang="En")
    assert len(voc2.voc) > len(voc.voc)


def test_create_text_model():
    voc = Vocabulary("191225.voc")
    nterms = len(voc.voc)
    tm = voc.create_text_model()
    assert len(tm["buenos dias"]) > 4
    assert nterms > tm.num_terms
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