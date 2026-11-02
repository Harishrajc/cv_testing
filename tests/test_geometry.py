from rigsafe.geometry import box_center, contains, normalize, point_in_polygon, sub_region

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
L_SHAPE = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]


def test_point_inside_and_outside_square():
    assert point_in_polygon((0.5, 0.5), SQUARE)
    assert not point_in_polygon((1.5, 0.5), SQUARE)
    assert not point_in_polygon((-0.1, 0.5), SQUARE)


def test_concave_polygon_notch_is_outside():
    assert point_in_polygon((0.5, 1.5), L_SHAPE)
    assert not point_in_polygon((1.5, 1.5), L_SHAPE)  # the cut-out corner


def test_normalize():
    assert normalize((960, 270), 1920, 1080) == (0.5, 0.25)


def test_box_center_and_contains():
    assert box_center((0, 0, 10, 20)) == (5, 10)
    assert contains((0, 0, 10, 10), (10, 10))
    assert not contains((0, 0, 10, 10), (11, 5))


def test_sub_region_top_band_with_padding():
    assert sub_region((0, 0, 100, 200), 0.0, 0.25, pad_x=0.1) == (-10, 0, 110, 50)


def test_overlap_of_smaller():
    from rigsafe.geometry import overlap_of_smaller
    assert overlap_of_smaller((0, 0, 10, 10), (2, 2, 4, 4)) == 1.0     # small fully inside
    assert overlap_of_smaller((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert overlap_of_smaller((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5
