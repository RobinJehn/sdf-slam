"""Unit tests for stability_ensemble: config rewriting and ensemble stats."""

import pytest

from stability_ensemble import jitter_config, summarize

BASE = """dataset_dir: data/x
output_dir: out/original
solver:
  backend: lm
  lambda_init: 5.0
  lambda_factor: 0.9
"""


def test_jitter_config_rewrites_lambda_and_output_dir():
    text = jitter_config(BASE, "/tmp/run_3", 1e-9)
    assert "output_dir: /tmp/run_3" in text
    assert "out/original" not in text
    line = next(l for l in text.splitlines() if "lambda_init" in l)
    value = float(line.split(":")[1])
    assert value == pytest.approx(5.0 * (1.0 + 1e-9), rel=0, abs=0)
    assert value != 5.0


def test_jitter_config_zero_jitter_keeps_value_exact():
    text = jitter_config(BASE, "/tmp/run_0", 0.0)
    line = next(l for l in text.splitlines() if "lambda_init" in l)
    assert float(line.split(":")[1]) == 5.0


def test_jitter_config_requires_lambda_init():
    with pytest.raises(ValueError):
        jitter_config("output_dir: out/x\n", "/tmp/run", 1e-9)


def test_summarize_counts_diverged_runs():
    stats = summarize([0.013, 0.014, 0.687, 0.013], divergence_factor=5.0)
    assert stats["best"] == pytest.approx(0.013)
    assert stats["diverged"] == 1
    assert stats["n"] == 4
    assert stats["spread"] == pytest.approx(0.687 - 0.013)
