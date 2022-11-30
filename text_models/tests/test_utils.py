# Copyright 2021 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np


def test_download_tokens():
    from text_models.utils import download_tokens
    from microtc.utils import load_model
    from os.path import isfile
    from os import unlink

    fname = download_tokens(dict(year=2020, month=2, day=14))
    assert isfile(fname)
    model = load_model(fname)
    print(model.most_common(10), model.update_calls)
    unlink(fname)
    fname = download_tokens(dict(year=2020, month=2, day=14), country="MX")
    assert isfile(fname)
    model2 = load_model(fname)
    assert len(model) != len(model2[0][1])
    unlink(fname)


def test_TStatistic():
    from text_models.utils import TStatistic
    from text_models import Vocabulary
    day = dict(year=2020, month=2, day=14)
    voc = Vocabulary(day, lang="En")
    tstats = TStatistic(voc.voc)
    value = tstats.compute("of~the")
    assert value > 316


def test_likelihood_ratios():
    from text_models.utils import LikelihoodRatios
    from text_models import Vocabulary
    day = dict(year=2020, month=2, day=14)
    voc = Vocabulary(day, lang="En")
    tstats = LikelihoodRatios(voc.voc)
    value = tstats.compute("of~the")
    assert tstats.compute("my~us") == 0
    # print(tstats.compute("imtreety_~the"), value)
    # assert tstats.compute("imtreety_~the") > 945
    # [tstats.compute(k) for k in voc if k.count("~")]
    assert value > 1503


def test_date_range():
    from text_models.utils import date_range

    init = dict(year=2020, month=2, day=1)
    end = dict(year=2020, month=3, day=1)
    lst = date_range(init, end)
    # print("**", len(lst), lst)
    assert len(lst) == 30
    end = lst[-1]
    assert end.year == 2020 and end.month==3 and end.day == 1
