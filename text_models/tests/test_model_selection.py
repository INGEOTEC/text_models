from text_models.model_selection import Node, NodeNB
from text_models.model_selection import ForwardSelection, BeamSelection
from EvoMSA.tests.test_base import get_data
import os


def test_node():
    n = Node([1, 2], models={k: None for k in range(4)},
             metric=lambda x, y: x)
    n2 = Node([1, 2], models={k: None for k in range(4)},
              metric=lambda x, y: x)
    assert hasattr(n, "_metric")
    s = set()
    s.add(n)
    s.add(n2)
    assert len(s) == 1
    neighbors = set(n)
    assert Node([1, 2, 3], metric=lambda x, y: x) in neighbors
    assert Node([1, 2, 0], metric=lambda x, y: x) in neighbors
    str(Node("213", None, metric=lambda x, y: x) == "1-2-3")


def test_node_model():
    n = Node([1, 2], models={1: ["a", "b"], 2: ["c", "d"]},
             metric=lambda x, y: x)
    for a, b in zip(n.model, ["ab", "cd"]):
        assert "".join(a) == b


def test_NB():
    from sklearn.metrics import f1_score
    X, y = get_data()
    models = {0: ["EvoMSA.model.Corpus", "sklearn.svm.LinearSVC"],
              1: ["b4msa.textmodel.TextModel", "EvoMSA.model.Bernulli"],
              2: ["EvoMSA.model.AggressivenessEs", "EvoMSA.model.Identity"]}
    a = NodeNB([0], models,
               metric=lambda y, hy: f1_score(y, hy, average="macro"))
    a.fit(X[:500], y[:500])
    perf = a.performance(X[500:], y[500:])
    print(perf)
    assert perf < 1 and perf > 0.1


def test_ForwardSelection():
    X, y = get_data()
    models = {0: ["EvoMSA.model.Corpus", "sklearn.svm.LinearSVC"],
              1: ["b4msa.textmodel.TextModel", "EvoMSA.model.Bernulli"],
              2: ["EvoMSA.model.AggressivenessEs", "EvoMSA.model.Identity"]}

    a = ForwardSelection(models,
                         node=NodeNB).fit(X[:500], y[:500],
                                          cache=os.path.join("tm", "fw"))
    a.run(X[500:], y[500:], cache=os.path.join("tm", "fw-test"))


def test_BeamSelection():
    X, y = get_data()
    models = {0: ["EvoMSA.model.Corpus", "sklearn.svm.LinearSVC"],
              1: ["b4msa.textmodel.TextModel", "EvoMSA.model.Bernulli"],
              2: ["EvoMSA.model.AggressivenessEs", "EvoMSA.model.Identity"]}

    a = BeamSelection(models,
                      node=NodeNB).fit(X[:500], y[:500],
                                       cache=os.path.join("tm", "fw"))
    a.run(X[500:], y[500:], cache=os.path.join("tm", "fw-test"),
          early_stopping=2)
