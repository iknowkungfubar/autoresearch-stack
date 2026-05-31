"""Extended tests for the report module.

Covers:
- Report class (sections, rendering, headers)
- generate_summary_report
- generate_comparison_report
- generate_full_report
"""


class TestReport:
    """Tests for the Report class."""

    def test_create_report(self):
        from report import Report

        r = Report("Test Report")
        assert r.title == "Test Report"

    def test_add_section(self):
        from report import Report

        r = Report("Test")
        r.add_section("Overview", "Test content")
        content = r.render()
        assert "Test content" in content

    def test_add_header(self):
        from report import Report

        r = Report("Test")
        r.add_header("Custom Header")
        content = r.render()
        assert "Custom Header" in content

    def test_multiple_sections(self):
        from report import Report

        r = Report("Multi")
        r.add_section("Section 1", "Content 1")
        r.add_section("Section 2", "Content 2")
        content = r.render()
        assert "Section 1" in content
        assert "Section 2" in content
        assert "Content 1" in content
        assert "Content 2" in content


class TestSummaryReport:
    """Tests for summary report generation."""

    def test_generate_summary_report(self):
        from report import generate_summary_report

        experiments = [
            {
                "id": 1,
                "change_description": "LR change",
                "change_type": "opt",
                "status": "kept",
                "val_bpb_after": 0.95,
            },
            {
                "id": 2,
                "change_description": "BS change",
                "change_type": "opt",
                "status": "reverted",
                "val_bpb_after": 1.2,
            },
        ]

        report = generate_summary_report(experiments)
        assert report is not None
        content = report.render()
        assert "kept" in content.lower()
        assert "reverted" in content.lower()

    def test_summary_report_empty(self):
        from report import generate_summary_report

        report = generate_summary_report([])
        assert report is not None
        content = report.render()
        assert "Summary" in content or "Experiment" in content

    def test_summary_report_single(self):
        from report import generate_summary_report

        experiments = [
            {
                "id": 1,
                "change_description": "Test",
                "change_type": "opt",
                "status": "kept",
                "val_bpb_after": 0.9,
            },
        ]

        report = generate_summary_report(experiments)
        content = report.render()
        assert "Test" in content


class TestComparisonReport:
    """Tests for comparison report generation."""

    def test_comparison_report(self):
        from report import generate_comparison_report

        experiment_sets = {
            "Set A": [
                {"id": 1, "change_description": "A1", "change_type": "opt",
                 "status": "kept", "val_bpb_after": 0.9},
                {"id": 2, "change_description": "A2", "change_type": "opt",
                 "status": "kept", "val_bpb_after": 0.95},
            ],
            "Set B": [
                {"id": 3, "change_description": "B1", "change_type": "opt",
                 "status": "reverted", "val_bpb_after": 1.1},
            ],
        }

        report = generate_comparison_report(
            experiment_sets, names=["First Run", "Second Run"]
        )
        assert report is not None
        content = report.render()
        assert "First Run" in content
        assert "Second Run" in content

    def test_comparison_report_single_set(self):
        from report import generate_comparison_report

        experiment_sets = {
            "Only Set": [
                {"id": 1, "change_description": "Test", "change_type": "opt",
                 "status": "kept", "val_bpb_after": 0.9},
            ],
        }

        report = generate_comparison_report(
            experiment_sets, names=["Only"]
        )
        content = report.render()
        assert "Only" in content


class TestFullReport:
    """Tests for full report generation."""

    def test_full_report_without_figures(self):
        from report import generate_full_report

        experiments = [
            {"id": 1, "change_description": "Test", "change_type": "opt",
             "status": "kept", "val_bpb_after": 0.9,
             "val_bpb_before": 1.0, "training_time": 5.0},
        ]

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_full_report(
                experiments=experiments,
                output_dir=tmpdir,
                baseline=1.0,
                include_figures=False,
            )
            assert isinstance(result, dict)
            assert "report" in result

    def test_full_report_empty(self):
        import tempfile

        from report import generate_full_report
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_full_report(
                experiments=[],
                output_dir=tmpdir,
                baseline=1.0,
                include_figures=False,
            )
            assert "report" in result

    def test_full_report_with_baseline(self):
        from report import generate_full_report

        experiments = [
            {"id": 1, "change_description": "Test", "change_type": "opt",
             "status": "kept", "val_bpb_after": 0.95,
             "val_bpb_before": 1.0, "training_time": 10.0},
        ]

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_full_report(
                experiments=experiments,
                output_dir=tmpdir,
                baseline=1.0,
            )
            path = result.get("report")
            assert path is not None or "report" in result
