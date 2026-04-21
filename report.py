"""
Report generation - Markdown experiment reports.

Phase 6: Reporting & Paper Generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass

# Optional imports for figures and statistics
if TYPE_CHECKING:
    from figures import FigureGenerator
    from stats import SummaryStatistics
else:
    try:
        from figures import FigureGenerator
        from stats import SummaryStatistics

        FIGURES_AVAILABLE = True
    except ImportError:
        FIGURES_AVAILABLE = False
        FigureGenerator = None
        SummaryStatistics = None


@dataclass
class ReportSection:
    """Report section."""

    title: str
    content: str
    level: int = 2


class Report:
    """Markdown report generator."""

    def __init__(self, title: str = "Experiment Report"):
        self.title = title
        self.sections: List[ReportSection] = []
        self.created_at = datetime.now().isoformat()

    def add_section(self, title: str, content: str, level: int = 2):
        """Add a section."""
        self.sections.append(ReportSection(title, content, level))

    def add_header(self, text: str):
        """Add a header."""
        self.sections.append(ReportSection(text, "", 1))

    def add_list(self, items: List[str]):
        """Add a list."""
        content = "\n".join(f"- {item}" for item in items)
        self.sections.append(ReportSection("", content, 0))

    def add_table(self, headers: List[str], rows: List[List[str]]):
        """Add a table."""
        # Header row
        content = "| " + " | ".join(headers) + " |\n"
        # Separator
        content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        # Data rows
        for row in rows:
            content += "| " + " | ".join(str(cell) for cell in row) + " |\n"

        self.sections.append(ReportSection("", content, 0))

    def add_code_block(self, code: str, language: str = "python"):
        """Add a code block."""
        content = f"```{language}\n{code}\n```"
        self.sections.append(ReportSection("", content, 0))

    def add_metrics_summary(self, stats: Dict[str, Any]):
        """Add metrics summary."""
        lines = [
            f"**Total Experiments:** {stats.get('total', 0)}",
            f"**Kept:** {stats.get('kept', 0)}",
            f"**Reverted:** {stats.get('reverted', 0)}",
            f"**Best val_bpb:** {stats.get('best_val_bpb', 'N/A')}",
            f"**Improvement:** {stats.get('improvement', 'N/A')}",
        ]
        self.add_section("Metrics Summary", "\n".join(lines))

    def render(self) -> str:
        """Render the report to markdown."""
        lines = []

        # Title
        lines.append(f"# {self.title}")
        lines.append(f"\n*Created: {self.created_at}*")
        lines.append("")

        # Sections
        for section in self.sections:
            if section.level == 1:
                lines.append(f"# {section.title}")
            elif section.level == 2:
                lines.append(f"## {section.title}")
            else:
                lines.append(section.title)

            if section.content:
                lines.append(section.content)

            lines.append("")

        return "\n".join(lines)

    def save(self, path: str):
        """Save report to file."""
        content = self.render()
        Path(path).write_text(content)
        return path


class ExperimentReport(Report):
    """Specialized experiment report."""

    def __init__(self, experiment_data: Dict[str, Any]):
        super().__init__(f"Experiment Report #{experiment_data.get('id', 'N/A')}")
        self.data = experiment_data
        self._build_report()

    def _build_report(self):
        """Build the report from experiment data."""
        # Header
        self.add_header(f"Experiment #{self.data.get('id', 'N/A')}")

        # Overview
        self.add_section(
            "Overview",
            f"""
**Change:** {self.data.get("change_description", "N/A")}
**Type:** {self.data.get("change_type", "N/A")}
**Status:** {self.data.get("status", "N/A")}
""",
        )

        # Results
        self.add_section(
            "Results",
            f"""
| Metric | Before | After |
|-------|--------|-------|
| val_bpb | {self.data.get("val_bpb_before", "N/A")} | {self.data.get("val_bpb_after", "N/A")} |
| Training Time | {self.data.get("training_time", "N/A")}s | - |
""",
        )

        if self.data.get("status") == "reverted":
            self.add_section(
                "Failure Analysis",
                f"""
**Classification:** {self.data.get("failure_classification", "N/A")}
**Diagnosis:** {self.data.get("failure_diagnosis", "N/A")}
""",
            )


def generate_summary_report(experiments: List[Dict]) -> Report:
    """Generate summary report from experiments."""
    report = Report("Autonomous Research Summary")

    # Calculate stats
    total = len(experiments)
    kept = sum(1 for e in experiments if e.get("status") == "kept")
    reverted = sum(1 for e in experiments if e.get("status") == "reverted")

    val_pb_values = [
        e.get("val_bpb_after", float("inf"))
        for e in experiments
        if e.get("status") == "kept"
    ]
    best = min(val_pb_values) if val_pb_values else None

    stats = {
        "total": total,
        "kept": kept,
        "reverted": reverted,
        "best_val_bpb": f"{best:.4f}" if best else "N/A",
    }

    # Add header
    report.add_header("Autonomous Research Summary")

    # Add metrics
    report.add_metrics_summary(stats)

    # Add experiment table
    if experiments:
        headers = ["ID", "Change", "Type", "Status", "val_bpb After"]
        rows = []
        for exp in experiments[-20:]:
            rows.append(
                [
                    str(exp.get("id", "")),
                    exp.get("change_description", "")[:30],
                    exp.get("change_type", ""),
                    exp.get("status", ""),
                    f"{exp.get('val_bpb_after', 'N/A')}",
                ]
            )
        report.add_table(headers, rows)

    return report


def generate_comparison_report(
    experiment_sets: Dict[str, List[Dict]],
    names: List[str],
) -> Report:
    """Generate comparison report."""
    report = Report("Experiment Comparison")

    report.add_header("Experiment Comparison")

    # Compare each set
    for name, experiments in zip(names, experiment_sets.keys()):
        section = f"### {name}\n"

        total = len(experiments)
        kept = sum(1 for e in experiments if e.get("status") == "kept")

        section += f"Total: {total}, Kept: {kept}, Rate: {kept / total * 100:.1f}%\n"

        report.add_section(name, section)

    return report


def generate_full_report(
    experiments: List[Dict],
    output_dir: str = "output",
    baseline: Optional[float] = None,
    include_figures: bool = True,
) -> Dict[str, str]:
    """
    Generate full report with statistics, figures, and markdown.

    Args:
        experiments: List of experiment dictionaries
        output_dir: Output directory for files
        baseline: Baseline metric for improvement calculation
        include_figures: Whether to generate figure plots

    Returns:
        Dictionary mapping file names to paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. Generate summary statistics
    if SummaryStatistics and baseline is not None:
        stats = SummaryStatistics(experiments, baseline=baseline)
        stats_json_path = output_path / "statistics.json"
        stats.to_json(str(stats_json_path))
        saved_files["statistics"] = str(stats_json_path)

        # Add stats to markdown report
        stats_summary = stats.summary_text()
    else:
        stats_summary = None

    # 2. Generate markdown report
    report = generate_summary_report(experiments)

    # Add stats section if available
    if stats_summary:
        report.add_section("Statistics Summary", stats_summary)

    # Add figures section
    if include_figures and FigureGenerator:
        try:
            gen = FigureGenerator()
            figure_paths = gen.generate_all_figures(
                experiments,
                str(output_path / "figures"),
                baseline=baseline or 1.0,
            )

            # Add figure references to markdown
            report.add_header("Figures")
            for name, path in figure_paths.items():
                rel_path = Path(path).relative_to(output_path)
                report.add_section(
                    name.replace("_", " ").title(),
                    f"![{name}]({rel_path})",
                )
                saved_files[name] = path
        except Exception as e:
            print(f"Warning: Could not generate figures: {e}")

    # Save markdown report
    md_path = output_path / "report.md"
    report.save(str(md_path))
    saved_files["report"] = str(md_path)

    return saved_files


if __name__ == "__main__":
    # Test
    print("Testing report generation...")

    # Sample data
    experiments = [
        {
            "id": 1,
            "change_description": "Increase LR",
            "change_type": "optimization",
            "status": "kept",
            "val_bpb_before": 1.0,
            "val_bpb_after": 0.95,
        },
        {
            "id": 2,
            "change_description": "Add dropout",
            "change_type": "architecture",
            "status": "reverted",
            "val_bpb_before": 0.95,
            "val_bpb_after": 0.97,
        },
        {
            "id": 3,
            "change_description": "Adjust warmup",
            "change_type": "curriculum",
            "status": "kept",
            "val_bpb_before": 0.95,
            "val_bpb_after": 0.92,
        },
    ]

    # Generate report
    report = generate_summary_report(experiments)

    # Save
    report.save("experiment_report.md")
    print(f"Report saved to experiment_report.md")
    print("\nReport content:")
    print(report.render()[:500] + "...")
