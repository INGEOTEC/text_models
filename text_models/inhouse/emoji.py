from microtc.emoticons import convert_emoji
from collections import defaultdict
from text_models.dataset import Dataset
from microtc.utils import save_model


def download(fname):
    from urllib import request
    import os
    output = fname.split("/")[-1]
    if os.path.isfile(output):
        return output
    request.urlretrieve(fname, output)
    return output


def remove_components(cdn):
    ll = []
    for d in cdn.split():
        if d in components:
            continue
        ll.append(d)
    return ' '.join(ll)


data = "https://www.unicode.org/Public/emoji/14.0/emoji-test.txt"    
data = map(lambda x: x.strip(), open('emoji-test.txt').readlines())
data = [x for x in data if len(x) and x[0] != '#']


emojis = defaultdict(list)
for line in data:
    emoji, desc = line.split(';')
    desc = desc.split('#')[0].strip()
    emojis[desc].append(emoji.strip())


emojis_filter = defaultdict(list)
components = set(emojis['component'])

for x in emojis['fully-qualified']:
    key = remove_components(x)
    emojis_filter[key].append(x)

components.add('FE0F')
m_qual = {remove_components(x): x for x in emojis_filter.keys()}

for x in emojis['minimally-qualified']:
    key = remove_components(x)
    value = m_qual[key]
    emojis_filter[value].append(x)

for x in emojis['unqualified']:
    key = remove_components(x)
    value = m_qual[key]
    emojis_filter[value].append(x)

output = dict()

for k, v in emojis_filter.items():
    ident = convert_emoji(k).strip()
    for item in v:
        output[convert_emoji(item).strip()] = ident


save_model(output, 'emojis.dict')
ds = Dataset()
ds.add(output)


ds.process('buenos xx 12 dias. {} todos! acción'.format(convert_emoji('1F44B 1F3FC')))