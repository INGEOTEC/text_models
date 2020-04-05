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
from text_models.place import Country


def test_country():
    cntr = Country()
    assert cntr.country("Estoy en mexico") == "MX"
    r = cntr.country("no se donde estoy")
    assert r is None
    r = cntr.country("georgia usa")
    assert r == "US"


def test_country_from_twitter():
    cntr = Country()
    r = cntr.country_from_twitter(dict(place=dict(country_code="Xx")))
    assert r == "Xx"
    r = cntr.country_from_twitter(dict(user=dict(location="morelia, mich., mexico")))
    assert r == "MX"
    r = cntr.country_from_twitter(dict(place=dict(country_code=""),
                                       user=dict(location="morelia, mich., mexico")))
    assert r == "MX"                                           


def test_bug_empty():
    cntr = Country()
    r = cntr.country("~sudsudan~")
    print("**", r)
    assert r is None or len(r) == 2