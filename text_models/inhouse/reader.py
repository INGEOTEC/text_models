# Copyright 2022 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from microtc.utils import tweet_iterator
from text_models.utils import get_text, handle_day
import datetime
from os.path import join
import gzip
from glob import glob


class TweetIteratorV1(object):
    def __init__(self, lang):
        self._lang = lang.lower().strip()

    def tweet_iterator(self, fname):
        lang = self._lang
        with gzip.open(fname) as fpt:
            while True:
                d = fpt.readline()
                if len(d) == 0:
                    break
                if len(d) == 1:
                    continue
                try:
                    _ = d.decode('utf-8').split('\t')[1]
                    try:
                        tw = json.loads(_)
                        if tw.get('lang', '') == lang:
                            try:
                                text = get_text(tw)
                            except KeyError:
                                text = tw['text']
                            if text[:2] == 'RT':
                                continue
                            tw['text'] = text
                            yield tw   
                    except json.decoder.JSONDecodeError:
                        continue
                except IndexError:
                    continue


class TweetIteratorV2(TweetIteratorV1):
    def tweet_iterator(self, fname):
        lang = self._lang        
        for tw in tweet_iterator(fname):
            if tw.get('lang', '') == lang:
                text = get_text(tw)
                if text[:2] == 'RT':
                    continue
                tw['text'] = text
                yield tw


class TweetIterator(object):
    def __init__(self, lang, path='raw') -> None:
        self.lang = lang
        self.path = path
        self._v1 = TweetIteratorV1(lang)
        self._v2 = TweetIteratorV2(lang)
        self.start_v2 = datetime.datetime(year=2021,
                                          month=6,
                                          day=1)

    def tweet_iterator(self, day) -> dict:
        day = handle_day(day)
        reader = self._v2 if day >= self.start_v2 else self._v1
        files = self.files(day)
        for file in files:
            for tweet in reader.tweet_iterator(file):
                yield tweet

    def files(self, day: datetime.datetime) -> list:
        lang = self.lang
        if day >= self.start_v2:
            date = join(str(day.year),
                        '{:02}'.format(day.month),
                        '{:02}'.format(day.day))
            init_geo_es_en = datetime.datetime(year=2022, month=5, day=12)
            if lang in ['es', 'en']:
                files = glob(join(self.path,
                                  f'{lang}-data',
                                  date, '*.json.gz'))
                if day == init_geo_es_en:
                    files.extend(glob(join(self.path,
                                           'GEO-es-en',
                                           date, '*.json.gz')))
                    files.extend(glob(join(self.path,
                                           'GEO',
                                           date, '*.json.gz')))
                elif day > init_geo_es_en:
                    files.extend(glob(join(self.path,
                                           'GEO-es-en',
                                           date, '*.json.gz')))
                else:
                    files.extend(glob(join(self.path,
                                           'GEO',
                                           date, '*.json.gz')))
            else:
               files = glob(join(self.path, 'GEO', date, '*.json.gz'))
            return files
        else:
            assert lang in ['ru', 'ar', 'es', 'en']
            path = join(self.path, '{}-prev-data'.format(lang))
            path = join(path, str(day.year)[-2:],
                        '{:02}'.format(day.month),
                        '{:02}'.format(day.day))
            return glob(join(path, '*.log.gz'))
