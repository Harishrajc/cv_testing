from rigsafe.detector import Detection
from rigsafe.ppe import COLOR_OK, COLOR_PERSON, COLOR_VIOLATION, is_violation, ppe_color, summarize


def det(label: str) -> Detection:
    return Detection(0, 0, 10, 10, 0.9, 0, label)


def test_violation_labels_are_red():
    assert is_violation(det("no_helmet"))
    assert ppe_color(det("no_helmet")) == COLOR_VIOLATION


def test_worn_ppe_is_green_and_person_has_own_color():
    assert not is_violation(det("helmet"))
    assert ppe_color(det("vest")) == COLOR_OK
    assert ppe_color(det("Person")) == COLOR_PERSON


def test_summarize_counts_people_and_violations():
    s = summarize([det("Person"), det("Person"), det("helmet"), det("no_helmet"), det("no_gloves")])
    assert s.people == 2
    assert s.violations == 2
    assert s.counts["helmet"] == 1


def test_hud_lines_list_missing_items():
    lines = summarize([det("Person"), det("no_helmet")]).hud_lines()
    assert lines[0] == "Workers: 1"
    assert "no_helmet:1" in lines[1]


def test_hud_lines_without_violations():
    assert summarize([det("Person")]).hud_lines()[1] == "Violations: 0"
