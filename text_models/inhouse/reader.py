import json
from microtc.utils import tweet_iterator
from text_models.utils import get_text
import gzip


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