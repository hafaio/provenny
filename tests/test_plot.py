# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeStubs=false
"""The layout output plugs into the common plotting libraries.

These are integration smoke tests: they build a figure from a diagram the way a caller
would and assert it renders/serializes without error, guarding the shape conventions
(radians vs degrees, semi- vs full-axis, boundary sampling) that plotting relies on.

The pyright directives above relax only the unknown-type rules: matplotlib and plotly
ship incomplete stubs, so strict mode floods this file with third-party type noise; real
type errors in the test's own code are still caught.
"""

from collections.abc import Collection, Mapping

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.patches import Circle, Ellipse, PathPatch
from matplotlib.path import Path

from provenny import proportional_venn

matplotlib.use("Agg")  # render off-screen; no display needed under pytest

_AREAS: Mapping[Collection[str], float] = {("A",): 1.0, ("B",): 1.0, ("A", "B"): 0.4}


def test_matplotlib_circle_patches(rng: np.random.Generator) -> None:
    """Circle layouts drop straight into matplotlib Circle patches with labels."""
    diagram = proportional_venn(_AREAS, rng=rng)
    figure, axes = plt.subplots()
    for name in diagram.names:
        circle = diagram[name]
        axes.add_patch(Circle(circle.center, circle.major, alpha=0.4))
    for names in ({"A"}, {"B"}, {"A", "B"}):
        subregion = diagram.zone(names)
        assert subregion is not None
        axes.annotate("".join(sorted(names)), subregion.center, ha="center")
    axes.set_aspect("equal")
    axes.autoscale()
    figure.canvas.draw()  # rasterizes; raises if any artist is malformed
    assert len(axes.patches) == len(diagram.names)
    plt.close(figure)


def test_matplotlib_ellipse_patches(rng: np.random.Generator) -> None:
    """Ellipse layouts map to matplotlib Ellipse patches (full axes, degrees)."""
    diagram = proportional_venn(_AREAS, mode="optimal", rng=rng)
    figure, axes = plt.subplots()
    for name in diagram.names:
        ellipse = diagram[name]
        axes.add_patch(
            Ellipse(
                ellipse.center,
                width=2.0 * ellipse.major,
                height=2.0 * ellipse.minor,
                angle=float(np.degrees(ellipse.angle)),
                alpha=0.4,
            )
        )
    axes.set_aspect("equal")
    axes.autoscale()
    figure.canvas.draw()
    assert len(axes.patches) == len(diagram.names)
    plt.close(figure)


def test_plotly_path_shapes(rng: np.random.Generator) -> None:
    """Ellipse outlines render as filled plotly path shapes that serialize."""
    diagram = proportional_venn(_AREAS, mode="optimal", rng=rng)
    figure = go.Figure()
    for name in diagram.names:
        figure.add_shape(type="path", path=diagram[name].svg_path(), opacity=0.4)
    both = diagram.zone({"A", "B"})
    assert both is not None
    center_x, center_y = both.center
    figure.add_annotation(x=center_x, y=center_y, text="A&B", showarrow=False)
    assert len(figure.to_dict()["layout"]["shapes"]) == len(diagram.names)
    assert figure.to_json()  # serializes for the browser without error


def test_plotly_fills_each_subregion(rng: np.random.Generator) -> None:
    """Each zone's boundary fills as its own plotly path shape -- per-region coloring."""
    diagram = proportional_venn(_AREAS, mode="optimal", rng=rng)
    figure = go.Figure()
    palette = ("red", "green", "blue")
    for names, color in zip(({"A"}, {"B"}, {"A", "B"}), palette, strict=True):
        subregion = diagram.zone(names)
        assert subregion is not None
        figure.add_shape(type="path", path=subregion.svg_path(), fillcolor=color)
    assert len(figure.to_dict()["layout"]["shapes"]) == len(palette)
    assert figure.to_json()


def test_matplotlib_fills_a_subregion(rng: np.random.Generator) -> None:
    """A zone boundary becomes a fillable matplotlib PathPatch."""
    diagram = proportional_venn(_AREAS, mode="optimal", rng=rng)
    figure, axes = plt.subplots()
    both = diagram.zone({"A", "B"})
    assert both is not None
    vertices, codes = both.matplotlib_path()
    axes.add_patch(PathPatch(Path(vertices, codes), facecolor="purple", alpha=0.5))
    axes.autoscale()
    figure.canvas.draw()
    assert len(axes.patches) == 1
    plt.close(figure)
