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
from .utils import download_geo, Gaussian, MobilityTransform, remove_outliers
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
    """ Taken from http://www.samuelbosch.com/2018/09/great-circle-calculations-with-numpy.html
    also available at: https://raw.githubusercontent.com/samuelbosch/blogbits/master/geosrc/numpy_greatcircle.py
    """
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


def location(x):
    """
    Location of a tweet. In the case, it is a bounding box
    the location is the average.

    :param x: Tweet
    :type x: dict
    :rtype: tuple
    """
    long_lat = x.get('coordinates', None)
    if long_lat is None:
        place = x["place"]
        bbox = place.get("bounding_box", dict()).get("coordinates")
        bbox = np.array([point(*x) for x in bbox[0]])
        b = bbox.mean(axis=0).tolist()            
    else:
        long_lat = long_lat.get("coordinates")
        b = point(*long_lat)
    return b


def _length(x):
    """
    Lenght between two points

    :param x: Two points
    :type x: list
    :rtype: float

    >>> from text_models.place import _length
    >>> points = [[-103.7420245057158, 17.915988111000047], [-100.1228494938091, 20.403147690813284]]
    >>> _length(points)
    470.06161477088165

    """

    points = [point(*_) for _ in x]
    uno = points[0]
    dos = points[1]
    return distance(uno[0], uno[1], dos[0], dos[1])


def length(x):
    """
    Bounding box length

    :param x: Tweet
    :type x: dict
    :rtype: float

    >>> from text_models.place import length
    >>> bbox = dict(place=dict(bounding_box=dict(coordinates=[[[-99.191996,19.357102],[-99.191996,19.404124],[-99.130965,19.404124],[-99.130965,19.357102]]])))
    >>> length(bbox)
    8.26567850078472

    """  
    long_lat = x.get('coordinates', None)
    if long_lat is not None:
        return 0

    place = x["place"]
    bbox = place.get("bounding_box", dict()).get("coordinates")[0]
    return _length([bbox[0], bbox[2]])


class CP(object):
    """
    Mexico Postal Codes
    
    >>> from text_models.place import CP
    >>> cp = CP()
    >>> tw = dict(coordinates=dict(coordinates=[-99.191996,19.357102]))
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

        b = location(x)
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


class Mobility(object):
    """
    Mobility on twitter

    :param day: Starting day default yesterday
    :type day: datetime
    :param window: Window used to perform the analysis
    :type window: int
    :param end: End of the period, use to override window.
    :type end: datetime

    >>> from text_models.place import Mobility
    >>> mobility = Mobility(window=5)
    >>> output = mobility.displacement(level=mobility.state)
    """

    def __init__(self, day=None, window=30, end=None):
        path = join(dirname(__file__), "data", "state.dict")
        self._states = load_model(path)
        path = join(dirname(__file__), "data", "bbox_country.dict")
        self._n_states = load_model(path)
        self._bbox = BoundingBox()
        self._dates = list()
        delta = datetime.timedelta(days=1)
        init = datetime.datetime(year=2015, month=12, day=16)
        day = self.__handle_day(day)
        if end is not None:
            end = self.__handle_day(end)
            window = (day - end).days + 1
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

    def __handle_day(self, day):
        """Inner function to handle the day
        
        :param day: day
        :type day: None | instance
        """

        delta = datetime.timedelta(days=1)
        if day is None:
            _ = time.localtime()
            day = datetime.datetime(year=_.tm_year,
                                    month=_.tm_mon,
                                    day=_.tm_mday) - delta
            return day
        if hasattr(day, "year") and hasattr(day, "month") and hasattr(day, "day"):
            return datetime.datetime(year=day.year,
                                     month=day.month,
                                     day=day.day)
        return day            

    @property
    def bounding_box(self):
        """Bounding box"""

        return self._bbox

    @property
    def dates(self):
        """Dates used on the analysis"""
        return self._dates

    @property
    def travel_matrices(self):
        """
        List of origin-destination matrix
        """

        return self._days

    def state(self, label, mex=False):
        """
        State that correspons to the label.

        :param label: Label of the point
        :type label: str
        :param mex_pc: Use Mexico's state identifier
        :type mex_pc: bool

        >>> from text_models.place import Mobility
        >>> mobility = Mobility(window=1)
        >>> mobility.state('MX:6435', mex=True)
        '16'
        >>> mobility.state("CA:12")
        'CA-ON'
        >>> mobility.state("MX:0")
        'MX-CHP'
        """

        xx = self._n_states.get(label[:2], set())
        if label in xx:
            return None
        
        if label[:2] == "MX" and mex:
            res = self.bounding_box.city(label)
            if res == label:
                return None
            return res[:2]
        try:
            return self._states[label]
        except KeyError:
            return None

    def country(self, key):
        """
        Country that correspond to the key.
        
        >>> from text_models.place import Mobility
        >>> mobility = Mobility(window=1)
        >>> mobility.country('MX:6435')
        'MX'
        
        """

        return key[:2]

    @staticmethod
    def fill_with_zero(output):
        """
        Fill mobility matrix with zero when a particular destination
        is not present.
        """

        todos = set()
        [todos.update(list(x.keys())) for x in output]
        O = defaultdict(list)
        for matriz in output:
            s = set(list(matriz.keys()))
            for x in todos - s:
                O[x].append(0)
            [O[k].append(v) for k, v in matriz.items()]
        return O
    
    def displacement(self, level=None):
        """
        Displacement matrix

        :param level: Aggregation function 
        """
        return self.overall(level=level)

    def overall(self, level=None):
        """
        Overall mobility, this counts for outward, inward and inside travels
        in the region of interest (i.e., level).

        :param level: Aggregation function
        """

        if level is None:
            level = self.country

        output = []
        for day in self.travel_matrices:
            matriz = Counter()
            for origen, destino in day.items():
                ori_code = level(origen)
                for dest, cnt in destino.items():
                    if ori_code is not None:
                        matriz.update({ori_code: cnt})
                    dest_code = level(dest)
                    if dest_code is not None and dest_code != ori_code:
                        matriz.update({dest_code: cnt})
            output.append(matriz)
        return self.fill_with_zero(output)

    def inside_mobility(self, level=None):
        """
        Mobility inside the region defined by level

        :param level: Aggregation function
        """

        if level is None:
            level = self.country

        output = []
        for day in self.travel_matrices:
            matriz = Counter()
            for origen, destino in day.items():
                ori_code = level(origen)
                if ori_code is None:
                    continue
                for dest, cnt in destino.items():
                    dest_code = level(dest)
                    if dest_code == ori_code:
                        matriz.update({ori_code: cnt})
            output.append(matriz)
        return self.fill_with_zero(output)

    def inside_outward(self, level):
        """
        Inside and outward mobility

        :param level: Aggregation function
        """

        output = []
        for day in self.travel_matrices:
            matrix = defaultdict(Counter)
            for origen, destino in day.items():
                ori_code = level(origen)
                if ori_code is None:
                    continue
                for dest, cnt in destino.items():
                    dest_code = level(dest)
                    if dest_code is None:
                        continue
                    travel = matrix[ori_code]
                    travel.update({dest_code: cnt})
            output.append(matrix)
        return output

    def outward(self, level=None):
        """
        Outward mobility in an origin-destination matrix
        
        :param level: Aggregation function
        :rtype: list
        """

        if level is None:
            level = self.country

        output = self.inside_outward(level=level)
        for data in output:
          for k, dest in data.items():
            if k in dest:
              del dest[k]
        return output

    def group_by_weekday(self, data):
        """
        Group the data by weekday works on a list of dictionaries
        where the value of the dictionary is a number.

        :param data: List of dictionaries, e.g., :py:func:`text_models.place.Mobility.inside_mobility`
        :type data: list
        :rtype: dict
        """
        output = dict()
        for key, values in data.items():
            weekday = defaultdict(list)
            [weekday[d.weekday()].append(v)
             for d, v in zip(self.dates, values)]
            output[key] = weekday
        return output

    def create_transform(self, data, transformation):
        """
        Instantiate the transform class
        """
        from .utils import MobilityException

        output = dict()
        for key, d in data.items():
            try:
                ins = transformation()
                ins.data = d
                output[key] = ins
            except MobilityException:
                continue
        return output

    def weekday_percentage(self, data):
        """
        Compute the percentage of each weekday using the median.

        :param data: Data, e.g., :py:func:`text_models.place.Mobility.displacement`
        :type data: dict
        :rtype: dict
        """

        weekday_data = self.group_by_weekday(data)
        return self.create_transform(weekday_data, MobilityTransform)

    def weekday_probability(self, data):
        """
        Normal distribution of weekday data.

        :param data: Data, e.g., :py:func:`text_models.place.Mobility.inside_mobility`
        :type data: dict
        :rtype: dict
        """
        class T(MobilityTransform):
            def transform(self, value):
                wdays = self._wdays
                value = np.atleast_1d(value)
                r = np.zeros(value.shape)
                for wd in range(7):
                    m = wdays == wd
                    r[m] = self.data[wd].predict_proba(value[m])
                return r

            @property
            def data(self):
                return self._data

            @data.setter
            def data(self, value):
                self._data = {k: Gaussian().fit(v) for k, v in value.items()}

        weekday_data = self.group_by_weekday(data)
        return self.create_transform(weekday_data, T)

    def cluster_percentage(self, data, n_clusters=None):
        """
        Compute the percentage using KMeans with K=7.

        :param data: Data, e.g., :py:func:`text_models.place.Mobility.inside_mobility`
        :type data: dict
        :rtype: dict
        :param n_clusters: Number of function to maximize
        :type n_clusters: int or function
        """

        class K(MobilityTransform):
            def transform(self, value):
                X = np.atleast_2d(value).T
                m = self.data.cluster_centers_[self.data.predict(X)]
                return (100 * (X - m) / m).flatten()

            @property
            def data(self):
                return self._data

            def n_clusters(self, X):
                from sklearn.metrics import silhouette_score
                from sklearn.cluster import KMeans
                from sklearn.utils import check_X_y
                if n_clusters is None:
                    return 7
                if isinstance(n_clusters, int):
                    return n_clusters
                perf = []
                for k in range(2, min(8, X.shape[0])):
                    km = KMeans(n_clusters=k).fit(X)
                    labels = km.predict(X)
                    _ = n_clusters(X, labels)
                    perf.append(_)
                return np.argmax(perf) + 2

            @data.setter
            def data(self, value):
                from sklearn.cluster import KMeans
                from .utils import MobilityException
                value = remove_outliers(value)
                if len(value) == 0:
                    raise MobilityException()
                X = np.atleast_2d(value).T
                n_clu = self.n_clusters(X)              
                self._data = KMeans(n_clusters=n_clu).fit(X)
        return self.create_transform(data, K)

    def transform(self, data, baseline):
        """
        Transform data using the baseline

        :param data: Mobility data
        :type data: dict
        :param baseline: Baseline used to compute the percentage, e.g., :py:func:`text_models.place.Mobility.median_weekday`
        :type baseline: dict

        """

        for v in baseline.values():
            v.mobility_instance = self
        return self._apply(data, baseline)

    @staticmethod
    def _apply(data, func):
        """
        :param data: Data to apply :py:attr:`func`
        :type data: dict
        :param func: Functions with method transform
        :type func: dict
        """

        return {k: func[k].transform(v) for k, v in data.items() if k in func}

Travel = Mobility

class BoundingBox(object):
    """
    The lowest resolution, on mobility, is the centroid of
    the bounding box provided by Twitter. Each centroid is
    associated with a label. This class provides the mapping
    between the geo-localization and the centroid's label. 
    """

    def __init__(self):
        path = join(dirname(__file__), "data", "bbox.dict")
        self._bbox = load_model(path)

    @property
    def bounding_box(self):
        """
        Bounding box data
        """

        return self._bbox

    @property
    def pc(self):
        """Postal code"""

        try:
            return self._cp
        except AttributeError:
            cp = CP()
            self._postal_code_names = cp.postal_code_names
            d = dict()
            for key, (lat, lon) in enumerate(self.bounding_box["MX"]):
                index = np.argmin(distance(cp.lat, cp.lon, lat, lon))
                d["MX:%s" % key] = cp.cp[index]
            self._cp = d
        return self._cp

    def city(self, label):
        """
        Mexico cities 
        """

        try:
            code = self.postal_code(label)
            data = self._postal_code_names[code]
            return "%s%s" % (data[0], data[2])
        except KeyError:
            return label

    def postal_code(self, label):
        """
        Mexico postal code given a label

        :param label: Bounding box label
        :type label: str

        >>> from text_models.place import BoundingBox
        >>> bbox = BoundingBox()
        >>> bbox.postal_code('MX:6435')
        '58000'

        """
        return self.pc[label]

    def label(self, data):

        """
        The label of the closest bounding-box centroid to the data

        :param data: A dictionary containing the country and the position
        :type data: dict

        >>> from text_models.place import BoundingBox
        >>> bbox = BoundingBox()
        >>> bbox.label(dict(country="MX", position=[0.34387610272769614, -1.76610232121455]))
        'MX:6435'

        """ 

        country = data["country"]
        try:
            bbox = self.bounding_box[country]
            pos = data["position"]
            mask = np.fabs(bbox - pos).sum(axis=1) == 0
            index = np.where(mask)[0]
            if index.shape[0] == 1:
                acl = "%s:%s" % (country, index[0])
            else:
                acl = np.argmin(distance(bbox[:, 0], bbox[:, 1],
                                         pos[0], pos[1]))
                acl = "%s:%s" % (country, acl)            
            return acl
        except KeyError:
            return country


class States(object):
    """
    Auxiliary function to retrieve 
    the States or Provinces geometries and attributes from Natural Earth.
    """
    def __init__(self):
        from cartopy.io import shapereader
        fname = shapereader.natural_earth(resolution='10m',
                                          category='cultural',
                                          name='admin_1_states_provinces')
        _ = shapereader.Reader(fname)
        self._records = {x.attributes["iso_3166_2"]: x for x in _.records()}

    def __getitem__(self, key):
        return self._records[key]

    def keys(self):
        return self._records.keys()

    def items(self):
        return self._records.items()

    def name(self, key):
        return self[key].attributes["name_en"]

    def associate(self, data, country=None):
        """
        Associate a array of points with the states. 

        :param data: Array of points in radians (lat, lon)
        :type data: list
        :param country: Country using two letters code
        :type country: str

        """
        from shapely.geometry import Point
        records = self._records
        data = np.rad2deg(data)
        data = data[:, ::-1]
        data = [[k, Point(a, b)] for k, (a, b) in enumerate(data)]
        country = country.upper()
        keys = {k for k in records.keys() if k[:2] == country}
        output = []
        for k in keys:
            if len(data) == 0:
                break
            d = records[k].geometry
            res = [[i, d.contains(x)] for i, x in data]
            output.extend([[i, k] for i, flag in res if flag])
            data = [d for d, (_, flag) in zip(data, res) if not flag]
        for i, point in data:
            _ = [[k, records[k].geometry.distance(point)] for k in keys]
            _.sort(key=lambda x: x[1])
            output.append([i, _[0][0]])
        output.sort(key=lambda x: x[0])
        return [o for _, o in output]
