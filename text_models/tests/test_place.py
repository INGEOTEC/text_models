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


def test_postal_code_names():
    from text_models.place import CP
    import os

    with open("t.cp", "w") as fpt:
        print("", file=fpt)
        print("d_codigo|c_estado|d_estado|c_mnpio|D_mnpio|otro", file=fpt)
        print("000|01|MM|010|NN|X", file=fpt)
        print("001|09|MM|010|NN|X", file=fpt)

    res = CP._postal_code_names("t.cp")
    assert res["001"] != "MM"
    os.unlink("t.cp")  


def test_travel():
    from text_models.place import Travel, CP
    travel = Travel(window=3)
    assert len(travel._days) == 3
    print(travel.num_users)


def test_travel_displacement():
    from text_models.place import Travel, CP
    travel = Travel(window=3)
    assert len(travel._days) == 3
    output = travel.displacement()
    for v in output.values():
        assert len(v) == 3


def test_travel_dates():
    from text_models.place import Travel, CP
    from datetime import datetime
    travel = Travel(datetime(year=2020, month=4, day=4),
                    window=4)
    print(travel.dates[-1].day, 4)
    assert travel.dates[-1].day == 4
    print(travel.dates[0].day, 1)
    assert travel.dates[0].day == 1


def test_travel_init():
    from text_models.place import Travel, CP
    from datetime import datetime
    travel = Travel(datetime(year=2015, month=12, day=18),
                    window=4)
    print(len(travel.dates))
    assert len(travel.dates) == 3


def test_utils_download():
    from text_models.utils import download
    try:
        download("fail")
    except Exception:
        return
    assert False


def test_travel_outward():
    from text_models.place import Travel
    from datetime import datetime
    travel = Travel(window=4)
    matrix = travel.outward(travel.country)
    assert matrix[-1]["MX"]


def test_bounding_box_label():
    from text_models.place import BoundingBox, CP

    bbox = BoundingBox()
    cp = CP()
    position = [cp.lat[0], cp.lon[0]]
    label = bbox.label(dict(country="MX", position=position))
    assert label[:2] == "MX"
    label = bbox.label(dict(country="xX", position=position))
    assert label == "xX"


def test_bounding_box_postal_code():
    from text_models.place import BoundingBox

    bbox = BoundingBox()
    label = bbox.label(dict(country="MX",
                            position=[0.34387610272769614,
                                      -1.76610232121455]))
    pc = bbox.postal_code(label)
    assert pc == "58000"


def test_bounding_box_city():
    from text_models.place import BoundingBox

    bbox = BoundingBox()
    city = bbox.city('MX:6435')
    assert city == "16053"
    city = bbox.city('MX:6435')
    assert city == "16053"
    city = bbox.city("US")
    assert city == "US"


def test_bounding_box_city_bug():
    from text_models.place import BoundingBox

    bbox = BoundingBox()
    code = "MX:8680"
    city = bbox.city(code)
    print(code, city)
    assert city != code


def test_travel_inside_mobility():
    from text_models.place import Travel
    from datetime import datetime
    travel = Travel(window=4)
    dis = travel.displacement(travel.country)
    inside = travel.inside_mobility(travel.country)
    assert sum(dis["MX"]) > sum(inside["MX"])


def test_travel_weekday():
    from text_models.place import Travel
    travel = Travel(window=21)
    inside = travel.inside_mobility(travel.country)
    baseline = travel.group_by_weekday(inside)
    for wk in range(7):
        _ = baseline["MX"][wk]
        print(_)
        assert sum(_) > 0


def test_travel_weekday_percentage():
    from text_models.place import Travel
    travel = Travel(window=21)
    inside = travel.inside_mobility(travel.country)
    baseline = travel.weekday_percentage(inside)    
    for wk in range(7):
        _ = baseline["MX"]
        print(_)
        assert _._data[wk] > 0
    for v in baseline.values():
        v.mobility_instance = travel
    y = baseline["MX"].transform(inside["MX"])
    assert len(y) == len(travel.dates)


def test_travel_percentage_by_weekday():
    from text_models.place import Travel
    import numpy as np
    travel = Travel(window=21)
    inside = travel.inside_mobility(travel.country)
    baseline = travel.weekday_percentage(inside)
    print(len(inside), inside["MX"])
    output = travel.transform(inside, baseline)
    for k in ["MX", "US"]:
        v = output[k]
        print(v)
        assert len([1 for x in v if x == 0]) == 7


def test_travel_percentage_by_weekday2():
    from text_models.place import Travel
    import numpy as np
    travel = Travel(window=21)
    inside = travel.inside_mobility(travel.bounding_box.city)
    baseline = travel.weekday_percentage(inside)
    print(len(baseline))
    assert len(baseline) < 14347 


def test_travel_weekday_probability():
    from text_models.utils import Gaussian
    from text_models.place import Travel
    import numpy as np
    travel = Travel(window=21)
    inside = travel.inside_mobility(travel.country)
    baseline = travel.weekday_probability(inside)
    print(baseline["MX"]._data[0])
    for wk in range(7):
        assert isinstance(baseline["MX"]._data[wk], Gaussian)
    output = travel.transform(inside, baseline)
    print(output["MX"])
    assert np.all(output["MX"] > 0.01)


def test_gaussian():
    import numpy as np
    from text_models.utils import Gaussian
    g = Gaussian().fit(np.random.random(100))
    print(g._mu, g._std)
    _ = g.predict_proba([0.1, 0.4, 0.41, 0.5, 10., -10])
    assert np.all(_[-2:] < 1e-6)
    assert np.all(_[:-2] > 0)
