# provenny

[![build](https://github.com/hafaio/provenny/actions/workflows/build.yml/badge.svg)](https://github.com/hafaio/provenny/actions/workflows/build.yml)
[![pypi](https://img.shields.io/pypi/v/provenny)](https://pypi.org/project/provenny/)
[![docs](https://img.shields.io/badge/docs-hafaio.github.io-blue)](https://hafaio.github.io/provenny/)

Lay out area-proportional euler/venn diagrams for any number of sets.

## Installation

```sh
pip install provenny  # or: uv add provenny
```

## Usage

Give the area of each subset by name and get back a layout:

```python
from provenny import proportional_venn

# Area of every subset you care about; omitted subsets are empty.
diagram = proportional_venn({"A": 1.0, "B": 1.0, "AB": 0.4})

diagram.names       # ('A', 'B')
diagram["A"]        # an Ellipse for set A (or the spelled-out diagram.ellipse("A"))

# A zone is a region inside some sets and outside the rest (None if it is empty)
ab = diagram.zone("AB")      # the A & B zone, or None
ab.center                    # (x, y) label point, always inside the zone
ab.area                      # exact realized area
ab.svg_path()                # boundary as a fillable path (see Plotting)
```

Each key is the collection of set names a subset is inside. For single-character names a
bare string is shorthand for its characters. You must use a tuple like `("Apple",)` for
multi-character names.

Set labels are not limited to strings -- any hashable object works (integers, enum
members, `frozenset`s), and the returned `Diagram` is generic over the label type:

```python
from enum import Enum, auto

class Team(Enum):
    RED = auto()
    BLUE = auto()

diagram = proportional_venn({(Team.RED,): 1.0, (Team.BLUE,): 1.0, (Team.RED, Team.BLUE): 0.4})
diagram.zone({Team.RED, Team.BLUE})
```

Ellipses fit targets circles cannot. `"ellipse"` keeps every pairwise overlap a single
connected lens -- the mode to reach for when you want ellipses. `"optimal"` drops that
constraint for the lowest area error it can reach, letting a pair overlap in two
disconnected lobes. Both stay circular when circles already suffice.

```python
diagram = proportional_venn({"A": 1.0, "B": 1.0, "AB": 0.4}, mode="ellipse")
```

### Plotting

`provenny` computes the layout and leaves rendering to your plotting library. Indexing a
diagram (or the equivalent `diagram.ellipse(name)`) gives an `Ellipse` that exports its
outline as a path in a few formats, so shapes drop into whatever you already use.

With `matplotlib`, build a patch from the named parameters (`Ellipse` wants *full* axes
and *degrees*, so double the semi-axes and convert the angle) and label the regions:

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

diagram = proportional_venn({"A": 1.0, "B": 1.0, "AB": 0.4}, mode="ellipse")
_, axes = plt.subplots()
for name in diagram.names:
    e = diagram[name]
    axes.add_patch(Ellipse(e.center, 2 * e.major, 2 * e.minor, angle=np.degrees(e.angle), alpha=0.4))
for names in ("A", "B", "AB"):
    z = diagram.zone(names)
    if z is not None:
        axes.annotate(names, z.center, ha="center")
axes.set_aspect("equal")
axes.autoscale()
```

For **plotly** (or bokeh, altair, raw svg, ...), `svg_path()` gives a cubic-Bézier path
string that fills as a shape:

```python
import plotly.graph_objects as go

figure = go.Figure()
for name in diagram.names:
    figure.add_shape(type="path", path=diagram[name].svg_path(), opacity=0.4)
```

Both `Ellipse` and `Zone` export three ways to access their boundary paths: `svg_path()`
(above), `matplotlib_path()` (a `(vertices, codes)` pair for `matplotlib.path.Path`), and
`sample(num)` (boundary points). So to fill each **subregion** a different color, path a
`Zone` instead of an `Ellipse`:

```python
for names, color in zip(("A", "B", "AB"), ("red", "green", "blue")):
    z = diagram.zone(names)
    if z is not None:
        figure.add_shape(type="path", path=z.svg_path(), fillcolor=color)
```

### Raw array interface

Passing a 1-D array of subset areas (indexed by subset bitmask minus one, bit `i` =
set `i`; for two sets `[|A|, |B|, |A & B|]`) returns the shapes directly:

```python
import numpy as np

from provenny import proportional_venn_array, zone

shapes = proportional_venn_array(np.array([1.0, 1.0, 0.4]))
# zone takes a boolean mask over the sets (inside[i] == inside set i); None if empty.
zone(shapes, np.array([True, True]))  # the A & B zone: .center, .area, .svg_path()
```
