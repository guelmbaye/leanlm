"""Raw model speed, and the two ways of getting it wrong.

Both were hit in practice: `llama-bench` without `-m` benchmarks llama.cpp's
default path rather than your model, and a resident `llama-server` holds the
weights in memory while competing for the cores the benchmark is timing.
"""
from __future__ import annotations

import stat as stat_module

import pytest

from leanlm.benchmarks.speed import TPS_REFERENCE, measure, render
from leanlm.shared.errors import LeanLMError

TABLE = (
    "| model | size | params | backend | threads | test | t/s |\n"
    "| qwen35 2B Q4_K - Medium | 1.18 GiB | 1.88 B | CPU | 4 | pp128 | 32.96 ± 1.01 |\n"
    "| qwen35 2B Q4_K - Medium | 1.18 GiB | 1.88 B | CPU | 4 | tg32 | 5.81 ± 0.66 |\n"
)


def _bench(tmp_path, body: str, exit_code: int = 0):
    fake = tmp_path / "llama-bench"
    fake.write_text(f"#!/usr/bin/env bash\ncat <<'OUT'\n{body}\nOUT\nexit {exit_code}\n",
                    encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat_module.S_IEXEC)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 40)
    return str(fake), str(model)


class TestMeasurement:
    def test_both_rows_are_parsed(self, tmp_path):
        binary, model = _bench(tmp_path, TABLE)
        result = measure(model, threads=4, binary=binary)
        kinds = {row["kind"]: row["tokens_per_second"] for row in result["rows"]}
        assert kinds["prompt processing"] == 32.96
        assert kinds["generation"] == 5.81

    def test_the_dispersion_is_kept(self, tmp_path):
        binary, model = _bench(tmp_path, TABLE)
        generation = next(r for r in measure(model, threads=4, binary=binary)["rows"]
                          if r["kind"] == "generation")
        assert generation["stdev"] == 0.66

    def test_the_model_is_always_named_on_the_command_line(self, tmp_path):
        """Omitting -m benchmarks llama.cpp's default path, which produces a
        confident number about a file you never chose."""
        binary, model = _bench(tmp_path, TABLE)
        assert f"-m {model}" in measure(model, threads=4, binary=binary)["command"]

    def test_a_missing_model_path_is_refused(self, tmp_path):
        binary, _ = _bench(tmp_path, TABLE)
        with pytest.raises(LeanLMError) as excinfo:
            measure("", threads=4, binary=binary)
        assert excinfo.value.record.code == "SPD-002"

    def test_a_missing_binary_names_the_provisioning_script(self):
        with pytest.raises(LeanLMError) as excinfo:
            measure("/tmp/m.gguf", threads=4, binary="llama-bench-absent")
        assert "provision" in excinfo.value.record.recommended_action


class TestScoreProjection:
    def test_the_throughput_component_uses_the_official_reference(self, tmp_path):
        binary, model = _bench(tmp_path, TABLE)
        result = measure(model, threads=4, binary=binary)
        assert result["throughput_component"] == round(5.81 / TPS_REFERENCE, 4)

    def test_reaching_the_reference_saturates(self, tmp_path):
        table = TABLE.replace("5.81 ± 0.66", "22.00 ± 1.00")
        binary, model = _bench(tmp_path, table)
        assert measure(model, threads=4, binary=binary)["throughput_component"] == 1.0

    def test_the_rendering_states_the_points_at_stake(self, tmp_path):
        binary, model = _bench(tmp_path, TABLE)
        rendered = render(measure(model, threads=4, binary=binary))
        assert "30 points" in rendered
        assert "15 tok/s" in rendered


class TestFailureModes:
    def test_an_unreadable_table_is_reported(self, tmp_path):
        binary, model = _bench(tmp_path, "something entirely different")
        with pytest.raises(LeanLMError) as excinfo:
            measure(model, threads=4, binary=binary)
        assert excinfo.value.record.code == "SPD-004"

    def test_a_failing_binary_is_reported_with_its_output(self, tmp_path):
        binary, model = _bench(tmp_path, "failed to load model", exit_code=1)
        with pytest.raises(LeanLMError) as excinfo:
            measure(model, threads=4, binary=binary)
        assert excinfo.value.record.code == "SPD-003"
        assert "failed to load" in excinfo.value.record.details["stderr"]

    def test_a_competing_server_is_named_in_the_output(self, tmp_path, monkeypatch):
        """It holds the model in memory and competes for the cores being timed."""
        import leanlm.benchmarks.speed as speed
        monkeypatch.setattr(speed, "competing_processes", lambda: ["llama-server.exe"])
        binary, model = _bench(tmp_path, TABLE)
        rendered = render(measure(model, threads=4, binary=binary))
        assert "llama-server.exe" in rendered
        assert "Stop it" in rendered


class TestEncodingAndFormatTolerance:
    """llama.cpp writes UTF-8; Windows decodes with the ANSI code page.

    Observed: `±` arriving as `Â±`, which made the row parser fail and report
    "no readable results" on output that was perfectly correct. The same
    mismatch corrupts any accented character in a generated answer, so the
    decoding is now pinned wherever a subprocess is read.
    """

    REAL = (
        "| model                    |     size |  params | backend | threads |"
        "  test |            t/s |\n"
        "| ------------------------ | -------: | ------: | ------- | ------: |"
        " ----: | -------------: |\n"
        "| qwen35 2B Q4_K - Medium  | 1.18 GiB |  1.88 B | CPU     |       4 |"
        " pp128 |   32.96 ± 1.01 |\n"
        "| qwen35 2B Q4_K - Medium  | 1.18 GiB |  1.88 B | CPU     |       4 |"
        "  tg32 |    5.81 ± 0.66 |\n"
    )

    def _rows(self, text: str):
        from leanlm.benchmarks.speed import _ROW, _STDEV, _VALUE
        parsed = []
        for match in _ROW.finditer(text):
            cell = match.group("cell")
            value = _VALUE.search(cell)
            spread = _STDEV.search(cell)
            parsed.append((match.group("test"), float(value.group("value")),
                           float(spread.group("stdev")) if spread else 0.0))
        return parsed

    def test_the_real_table_is_parsed(self):
        assert self._rows(self.REAL) == [("pp128", 32.96, 1.01),
                                         ("tg32", 5.81, 0.66)]

    def test_a_mis_decoded_separator_does_not_break_it(self):
        """The parser must not be hostage to one character surviving a decode."""
        assert self._rows(self.REAL.replace("±", "Â±")) == [
            ("pp128", 32.96, 1.01), ("tg32", 5.81, 0.66)]

    def test_a_missing_deviation_is_zero_not_invented(self):
        """A looser pattern read "32.96" as 32 plus a deviation of 96 --
        inventing a dispersion the benchmark never reported."""
        table = self.REAL.replace(" ± 1.01", "").replace(" ± 0.66", "")
        assert self._rows(table) == [("pp128", 32.96, 0.0), ("tg32", 5.81, 0.0)]

    def test_integer_throughput_is_accepted(self):
        assert self._rows(self.REAL.replace("32.96 ± 1.01", "33")) [0] == (
            "pp128", 33.0, 0.0)

    def test_utf8_is_pinned_wherever_a_subprocess_is_read(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2] / "src" / "leanlm"
        for module in ("benchmarks/speed.py", "packages/inference/backends.py"):
            source = (root / module).read_text(encoding="utf-8")
            for index, line in enumerate(source.splitlines()):
                if "text=True" in line and "include_text" not in line:
                    window = "\n".join(source.splitlines()[index:index + 4])
                    assert 'encoding="utf-8"' in window, f"{module}:{index + 1}"
