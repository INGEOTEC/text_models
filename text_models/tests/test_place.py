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


# if __name__ == "__main__":
#     test_bounding_box_city()