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
import time
import datetime
from .utils import download_geo
from collections import defaultdict, Counter
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
    >>> cp.convert(tw)
    '01040'
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

    def postal_code(self, lat, lon, degrees=True):
        """
        Postal code

        :param lat: Latitude
        :type lat: float
        :param lon: Longitude
        :type lon: float
        :param degrees: Indicates whether the point is in degrees
        :type degrees: bool

        >>> from text_models.place import CP
        >>> cp = CP()
        >>> cp.postal_code(19.357102, -99.191996)
        '01040'
        """
        if degrees:
            lat, lon = point(lon, lat)
        d = distance(self.lat, self.lon, lat, lon)
        return self.cp[np.argmin(d)]

    def convert(self, x):
        """
        Obtain the postal code from a tweet

        :param x: Tweet
        :type x: dict
        :return: Postal Code
        :rtype: str
        """
        long_lat = x.get('coordinates', None)
        if long_lat is None:
            place = x["place"]
            bbox = place.get("bounding_box", dict()).get("coordinates")
            bbox = np.array([point(*x) for x in bbox[0]])
            b = bbox.mean(axis=0).tolist()            
        else:
            b = point(*long_lat)
        return self.postal_code(b[0], b[1], degrees=False)

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
        """
        Dictionary containing a descripcion of a postal code

        >>> from text_models.place import CP
        >>> cp = CP()
        >>> cp.postal_code_names["58000"]
        ['16', 'Michoacán de Ocampo', '053', 'Morelia']
        """
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
        'Aguascalientes'
        """

        return self.postal_code_names[postal_code][1]


class Travel(object):
    """
    Mobility on twitter

    :param day: Starting day default yesterday
    :type day: datetime
    :param window: Window used to perform the analysis
    :type window: int

    >>> from text_models.place import Travel
    >>> travel = Travel(window=5)
    >>> output = travel.displacement(level=travel.state)
    """

    def __init__(self, day=None, window=30):
        self._cp = CP()
        self._dates = list()
        delta = datetime.timedelta(days=1)
        init = datetime.datetime(year=2015, month=12, day=16)
        if day is None:
            _ = time.localtime()
            day = datetime.datetime(year=_.tm_year,
                                    month=_.tm_mon,
                                    day=_.tm_mday) - delta
                                           
        days = []
        while len(days) < window and day >= init:
            try:
                fname = download_geo("%s%02i%02i.travel" % (str(day.year)[-2:],
                                                            day.month,
                                                            day.day))
            except Exception:
                day = day - delta
                continue
            self._dates.append(day)
            day = day - delta
            days.append(load_model(fname))
        self._days = [x for x, _ in days]
        self.num_users = [x for _, x in days]
        self._days.reverse()
        self.num_users.reverse()
        self._dates.reverse()

    @property
    def dates(self):
        """Dates used on the analysis"""
        return self._dates

    def state(self, key):
        """
        State that correspons to the postal code.
        It works only for Mexico.

        >>> from text_models.place import Travel
        >>> travel = Travel(window=1)
        >>> travel.state("20900")
        '01'
        """
        if len(key) == 2:
            return None
        return self._cp.postal_code_names[key][0]


    def country(self, key):
        """
        Country that correspond to the key.
        
        >>> from text_models.place import Travel
        >>> travel = Travel(window=1)
        >>> travel.country("20900")
        'MX'
        >>> travel.country("US")
        'US'
        
        """

        if len(key) == 5:
            return "MX"
        return key
    
    def displacement(self, level=None):
        """
        Displacement matrix

        :param level: Aggregation function 
        """

        if level is None:
            level = self.state

        output = []
        for day in self._days:
            matriz = Counter()
            for origen, destino in day.items():
                for dest, cnt in destino.items():
                    ori_code = level(origen)
                    if ori_code is not None:
                        matriz.update({ori_code: cnt})
                    dest_code = level(dest)
                    if dest_code is not None and dest_code != ori_code:
                        matriz.update({dest_code: cnt})
            output.append(matriz)
        todos = set()
        [todos.update(list(x.keys())) for x in output]
        O = defaultdict(list)
        for matriz in output:
            s = set(list(matriz.keys()))
            for x in todos - s:
                O[x].append(0)
            [O[k].append(v) for k, v in matriz.items()]
        return O

