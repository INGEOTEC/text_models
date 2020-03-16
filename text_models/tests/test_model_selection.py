from text_models.model_selection import Node
from text_models.model_selection import ForwardSelection, BeamSelection
from EvoMSA.tests.test_base import get_data
import os
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


def test_node():
    kwargs = dict(metric=lambda y, hy: f1_score(y, hy, average="macro"),
                  split_dataset=KFold(n_splits=2,
                                      random_state=1,
                                      shuffle=True),
                  aggregate=lambda x: x,
                  cache=os.path.join("tm", "NB"))
    n = Node([1, 2], models={k: None for k in range(4)},
             **kwargs)
    n2 = Node([1, 2], models={k: None for k in range(4)},
              **kwargs)
    assert hasattr(n, "_metric")
    s = set()
    s.add(n)
    s.add(n2)
    assert len(s) == 1
    neighbors = set(n)
    assert Node([1, 2, 3], **kwargs) in neighbors
    assert Node([1, 2, 0], **kwargs) in neighbors
    str(Node("213", None, **kwargs) == "1-2-3")


def test_node_model():
    kwargs = dict(metric=lambda y, hy: f1_score(y, hy, average="macro"),
                  split_dataset=KFold(n_splits=2,
                                      random_state=1,
                                      shuffle=True),
                  aggregate=lambda x: x,
                  cache=os.path.join("tm", "NB"))
    
    n = Node([1, 2], models={1: ["a", "b"], 2: ["c", "d"]},
             **kwargs)
    for a, b in zip(n.model, ["ab", "cd"]):
        assert "".join(a) == b


def test_node_performance():
    from EvoMSA.utils import download
    
    X, y = get_data()
    kf = KFold(n_splits=2, random_state=1, shuffle=True)
    models = {0: [download("b4msa_Es.tm"), "sklearn.svm.LinearSVC"],
              1: ["b4msa.textmodel.TextModel", "EvoMSA.model.Bernulli"],
              2: ["EvoMSA.model.AggressivenessEs", "EvoMSA.model.Identity"]}
    a = Node([0], models,
             metric=lambda y, hy: f1_score(y, hy, average="macro"),
             split_dataset=kf,
             aggregate=lambda x: x,
             cache=os.path.join("tm", "NB"))
    # a.fit(X[:500], y[:500])
    perf = a.performance(X, y)
    assert len(perf) == 2
    print(perf)
    perf = np.mean(perf)
    assert perf < 1 and perf > 0.40


def test_ForwardSelection():
    X, y = get_data()
    models = {0: ["EvoMSA.model.Corpus", "sklearn.svm.LinearSVC"],
              1: ["b4msa.textmodel.TextModel", "EvoMSA.model.Bernulli"],
              2: ["EvoMSA.model.AggressivenessEs", "EvoMSA.model.Identity"]}
    a = ForwardSelection(models, output="fw.node")
    node = a.run(X, y)
    assert isinstance(node, Node)
    assert node.perf > 0 and node.perf < 1
    assert os.path.isfile("fw.node")


def test_BeamSelection():
    X, y = get_data()
    models = {0: ["EvoMSA.model.Corpus", "sklearn.svm.LinearSVC"],
              1: ["b4msa.textmodel.TextModel", "EvoMSA.model.Bernulli"],
              2: ["EvoMSA.model.AggressivenessEs", "EvoMSA.model.Identity"]}

    a = BeamSelection(models, output="beam.node")
    node = a.run(X, y, early_stopping=2)
    assert isinstance(node, Node)
    assert node.perf > 0 and node.perf < 1
    assert os.path.isfile("beam.node")
