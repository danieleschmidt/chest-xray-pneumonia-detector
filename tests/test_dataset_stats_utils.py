from src.dataset_stats import _sort_items, print_stats


def test_sort_items_by_count():
    counts = {"a": 1, "b": 3, "c": 2}
    items = _sort_items(counts, "count")
    assert items[0][0] == "b"
    assert items[-1][0] == "a"


def test_print_stats(capsys):
    counts = {"x": 1, "y": 1}
    print_stats(counts)
    captured = capsys.readouterr().out
    assert "Total images: 2" in captured
