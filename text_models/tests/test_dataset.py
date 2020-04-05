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
from text_models.dataset import Dataset


def test_dataset():
    from microtc.utils import load_model
    from EvoMSA.utils import download

    tm = load_model(download("b4msa_Es.tm"))
    dset = Dataset(lang="Es")
    for a, b in zip(dset.textModel["buenos"],
                    tm["buenos"]):
        assert a[0] == b[0] and a[1] == b[1]


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
    dset = Dataset()
    dset.add(dset.load_emojis())
    dset.add(dset.tm_words())
    xx = dset.process("xxx good 9 morning xxx fax x la", "~x~")
    for a, b in zip(xx, ["~xxx~good~9~morning~xxx~fax~", "~la~", "~la~"]):
        print(a, b)
        assert a == b
    xx = dset.process("xxx good 9 morning xxx fax x la", "9")
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
    dset = load_model(download("country.ds"))
    r = dset.klass("mexico y usa")
    assert len(r.intersection(set(["US", "MX"]))) == 2