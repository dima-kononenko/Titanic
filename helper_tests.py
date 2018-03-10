import helpers as h
import pandas as pd

#test 1
m, c = h.map_category(pd.Series(["A", "A", "A"]))

assert m == {"A": 1}
assert not (c - pd.Series([1, 1, 1])).all()

#test 2
m, c = h.map_category(pd.Series(["A", "A", "B"]))

assert m == {"A": 1, "B": 2}
assert not (c - pd.Series([1, 1, 2])).all()

#test 3
m, c = h.map_category(pd.Series(["A", "A", "B", None]))

assert m == {"A": 1, "B": 2, None: 3}
assert not (c - pd.Series([1, 1, 2, 3])).all()

#test 4
m, c = h.map_category(pd.Series(["B", "A", "B", None]), {"A": 1, "B": 2})
assert m == {"A": 1, "B": 2, None: 3}
assert not (c - pd.Series([1, 1, 2, 3])).all()