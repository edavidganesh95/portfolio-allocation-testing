from src.config import load_reference_example, load_universe, parse_ticker_input


def test_universe_has_no_personal_portfolio_state():
    cfg = load_universe()
    assert not hasattr(cfg, "current_portfolio")
    assert "TMFC" in cfg.assets


def test_reference_example_sums_to_one():
    ref = load_reference_example()
    assert abs(sum(ref.weights.values()) - 1.0) < 1e-9


def test_parse_ticker_input_accepts_multiple_separators_and_deduplicates():
    raw = "qqqm, IJR\nVXUS; qqqm 2800.hk"
    assert parse_ticker_input(raw) == ["QQQM", "IJR", "VXUS", "2800.HK"]
