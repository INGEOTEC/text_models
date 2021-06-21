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


from scipy.stats.stats import mode


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