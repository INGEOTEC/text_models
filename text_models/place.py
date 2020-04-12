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
import numpy as np
from os.path import join, dirname
EARTH_RADIUS = 6371.009


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


def distance(lat1, lng1, lat2, lng2):
    # lat1, lng1 = a
    # lat2, lng2 = b
    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    d = np.arctan2(np.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                           (cos_lat1 * sin_lat2 -
                            sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                   sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

    return EARTH_RADIUS * d


def point(longitude, latitude):
    # longitude, latitude = x['lat'], x['long']
    if latitude < -90:
        latitude = -180 - latitude
        longitude = longitude + 180
    if latitude > 90:
        latitude = 180 - latitude
        longitude = longitude + 180
    return np.radians(latitude), np.radians(longitude)


class CP(object):
    """
    Mexico Postal Codes
    
    >>> from text_models.place import CP
    >>> cp = CP()
    >>> tw = dict(coordinates=[-99.191996,19.357102])
    >>> codigo = cp.convert(tw)
    >>> box = dict(place=dict(bounding_box=dict(coordinates=[[[-99.191996,19.357102],[-99.191996,19.404124],[-99.130965,19.404124],[-99.130965,19.357102]]])))
    >>> cp.convert(box)
    '03100'
    """
    def __init__(self):
        path = join(dirname(__file__), "data", "CP.info")
        m = load_model(path)
        self.cp = [x[0] for x in m]
        self.lat = np.radians([x[1] for x in m])
        self.lon = np.radians([x[2] for x in m])

    def convert(self, x):
        """
        Obtain the postal code from a tweet

        :param x: Tweet
        :type x: dict
        :return: Postal Code
        :rtype: str
        """
        lat, lon = self.lat, self.lon
        cluster = self.cp
        long_lat = x.get('coordinates', None)
        if long_lat is None:
            place = x["place"]
            bbox = place.get("bounding_box", dict()).get("coordinates")
            bbox = np.array([point(*x) for x in bbox[0]])
            b = bbox.mean(axis=0).tolist()            
        else:
            b = point(*long_lat)
        d = distance(lat, lon, b[0], b[1])
        return cluster[np.argmin(d)]

    @staticmethod
    def _postal_code_names(path):
        # path = join(dirname(__file__), "data", "CP.txt")
        with open(path, encoding="latin-1", errors="ignore") as fpt:
            lines = fpt.readlines()
        lines = [x.split("|") for x in lines[1:]]
        header = {v: k for k, v in enumerate(lines[0])}
        pc_names = dict()
        for line in lines[1:]:
            code = line[header["d_codigo"]]
            state_c = line[header["c_estado"]]
            if state_c == "09":
                state = "Ciudad de México"
            else:
                state = line[header["d_estado"]]
            mun_c = line[header["c_mnpio"]]
            mun = line[header["D_mnpio"]]
            data = [state_c, state, mun_c, mun]
            if code in pc_names:
                print("Error: (%s) %s - %s" % (code, pc_names[code], data))
            pc_names[code] = data
        return pc_names

    @property
    def postal_code_names(self):
        try:
            return self._pc_names
        except AttributeError:
            path = join(dirname(__file__), "data", "CP.desc")
            self._pc_names = load_model(path)
        return self._pc_names

    def state(self, postal_code):
        """
        >>> from text_models.place import CP
        >>> cp = CP()
        >>> cp.state("20900")
        """

        return self.postal_code_names[postal_code][1]

