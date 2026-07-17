"""Test the named (dict-input) proportional_venn interface."""

import numpy as np
import pytest

from provenny import Diagram, Ellipse, proportional_venn, proportional_venn_array


def test_named_input_returns_a_diagram(rng: np.random.Generator) -> None:
    """A mapping of named subsets yields a Diagram of names and shapes."""
    diagram = proportional_venn(
        {("A",): np.pi, ("B",): np.pi, ("A", "B"): 0.6}, rng=rng
    )
    assert isinstance(diagram, Diagram)
    assert diagram.names == ("A", "B")
    assert diagram.shapes.shape == (2, 5)  # circles are the equal-axis ellipse case
    # diagram[name] wraps that set's row as an Ellipse
    assert np.allclose(diagram["A"].array, diagram.shapes[0])
    assert np.allclose(diagram["B"].array, diagram.shapes[1])


def test_diagram_is_hashable(rng: np.random.Generator) -> None:
    """A Diagram hashes and compares without raising on its shapes ndarray (identity semantics)."""
    diagram = proportional_venn(
        {("A",): np.pi, ("B",): np.pi, ("A", "B"): 0.6}, rng=rng
    )
    assert isinstance(hash(diagram), int)  # no "unhashable ndarray"
    assert diagram == diagram  # noqa: PLR0124 -- reflexive equality must not raise
    assert diagram in {diagram}


def test_named_matches_array_ordering() -> None:
    """The named path builds the same bitmask-ordered areas as the raw array path."""
    named = proportional_venn(
        {("A",): np.pi, ("B",): np.pi, ("A", "B"): 0.6}, rng=np.random.default_rng(0)
    )
    array = proportional_venn_array(
        np.array([np.pi, np.pi, 0.6]), rng=np.random.default_rng(0)
    )
    assert np.allclose(named.shapes, array)


def _in_circle(shape: Ellipse, point: tuple[float, float]) -> bool:
    center_x, center_y = shape.center
    return bool(
        (point[0] - center_x) ** 2 + (point[1] - center_y) ** 2 <= shape.major**2
    )


def test_zone_labels_the_right_region(rng: np.random.Generator) -> None:
    """A zone's center lands inside exactly the named sets."""
    diagram = proportional_venn(
        {("A",): np.pi, ("B",): np.pi, ("A", "B"): 0.6}, rng=rng
    )
    both = diagram.zone({"A", "B"})
    a_only = diagram.zone({"A"})
    assert both is not None and a_only is not None
    assert _in_circle(diagram["A"], both.center) and _in_circle(
        diagram["B"], both.center
    )
    assert _in_circle(diagram["A"], a_only.center)
    assert not _in_circle(diagram["B"], a_only.center)


def test_zone_empty_unknown_and_exterior(rng: np.random.Generator) -> None:
    """An empty zone is None; an unknown name raises KeyError; no names raises ValueError."""
    diagram = proportional_venn({("A",): np.pi, ("B",): np.pi}, rng=rng)  # no A&B
    assert diagram.zone({"A", "B"}) is None
    with pytest.raises(KeyError):
        diagram.zone({"A", "Z"})
    with pytest.raises(ValueError, match="exterior"):
        diagram.zone(set())


def test_named_ellipse_mode(rng: np.random.Generator) -> None:
    """Named input works with the ellipse modes too, and gives the same canonical rows."""
    diagram = proportional_venn(
        {("A",): np.pi, ("B",): np.pi, ("A", "B"): 0.6}, mode="optimal", rng=rng
    )
    assert diagram.shapes.shape == (2, 5)
    assert np.all(diagram.shapes[:, 2] >= diagram.shapes[:, 3])  # major first


def test_single_char_string_keys(rng: np.random.Generator) -> None:
    """A bare string key is a collection of single-character set names."""
    diagram = proportional_venn({"A": np.pi, "B": np.pi, "AB": 0.6}, rng=rng)
    assert diagram.names == ("A", "B")
    both = diagram.zone("AB")  # string works here too, as {"A", "B"}
    assert both is not None
    assert _in_circle(diagram["A"], both.center) and _in_circle(
        diagram["B"], both.center
    )


def test_ellipse_method_matches_indexing(rng: np.random.Generator) -> None:
    """diagram.ellipse(name) is the spelled-out diagram[name], KeyError and all."""
    diagram = proportional_venn({"A": np.pi, "B": np.pi, "AB": 0.6}, rng=rng)
    assert diagram.ellipse("A") == diagram["A"]
    assert [e.center for e in map(diagram.ellipse, diagram)] == [
        diagram[name].center for name in diagram.names
    ]
    with pytest.raises(KeyError):
        diagram.ellipse("Z")


def test_diagram_mapping_behaviors(rng: np.random.Generator) -> None:
    """A diagram acts like a mapping of names: len, iter, in, and KeyError."""
    diagram: Diagram[str] = proportional_venn({("A",): np.pi, ("B",): np.pi}, rng=rng)
    assert len(diagram) == len(diagram.names)
    assert list(diagram) == ["A", "B"]
    assert "A" in diagram and "Z" not in diagram
    with pytest.raises(KeyError):
        _ = diagram["Z"]


def test_arbitrary_hashable_labels(rng: np.random.Generator) -> None:
    """Sets can be labeled with any hashable object, not just strings."""
    red, blue = (1, 0, 0), (0, 0, 1)  # arbitrary hashable labels (RGB tuples)
    diagram = proportional_venn(
        {(red,): np.pi, (blue,): np.pi, (red, blue): 0.6}, rng=rng
    )
    assert set(diagram.names) == {red, blue}
    assert np.allclose(diagram[red].array, diagram.shapes[diagram.names.index(red)])
    both = diagram.zone({red, blue})  # exists and has a label point
    assert both is not None
    assert np.isfinite(both.center).all()


def test_diagram_rejects_mismatched_shapes() -> None:
    """A Diagram must carry one canonical ellipse row per name."""
    rows = np.zeros((2, 5))
    rows[:, 2:4] = 1.0
    with pytest.raises(ValueError, match="row per name"):
        Diagram(("A", "B", "C"), rows)  # three names, two rows
    with pytest.raises(ValueError, match="row per name"):
        Diagram(("A", "B"), np.zeros((2, 3)))  # circle rows, not ellipse rows
