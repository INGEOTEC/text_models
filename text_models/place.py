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
from EvoMSA.utils import download
from microtc.utils import load_model


class Country(object):
    """
    Obtain the country from a text.

    >>> from text_models.place import Country
    >>> cntr = Country()
    >>> cntr.country("I live in Mexico.")
    'MX'
    """
    def __init__(self):
        self._country = load_model(download("country.ds"))
        self._location = load_model(download("country-loc.ds"))

    def country(self, text):
        """
        Identify a country in a text

        :param text: Text
        :type text: str
        :return: The two letter country code
        """

        res = self._country.klass(text)
        if len(res) == 1:
            cc = list(res)[0]
            if len(cc) == 2:
                return cc
        res2 = self._location.klass(text)
        if len(res2) == 1:
            cc = list(res2)[0]
            if len(cc) == 2:
                return cc
        r = res.intersection(res2)
        if len(r) == 1:
            cc = list(r)[0]
            if len(cc) == 2:
                return cc
        return None


    def country_from_twitter(self, tw):
        """
        Identify the country from a tweet.

        :param tw: Tweet
        :type tw: dict
        """
        place = tw.get("place", None)
        if place is None:
            return self.country(tw["user"]["location"])
        country_code = place.get("country_code")
        if country_code is None or len(country_code) < 2:
            return self.country(tw["user"]["location"])
        return country_code
