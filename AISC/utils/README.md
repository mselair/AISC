# utils
Features and utilities which are nice to have working not only with this project.

### feature_util
Machine learning tools for feature augmentation, finding outliers in data, z-score normalization, balancing classes etc.

### types

#### ObjDict
Combines matlab struct and python dict. Mirroring keys and attributes and it's values.
Nested initialization implemented as well.

```python
from AISC.utils.types import ObjDict

x = ObjDict()
x['a'] = 1
x.b = 2
print(x['a'], x.a)
print(x['b'], x.b)

del x.b
print(x)
del x['a']

x.a.b.c = 10 # creates nested ObjDicts
x['b']['b']['c'] = 20

``` 


