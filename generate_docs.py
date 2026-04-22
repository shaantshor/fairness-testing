"""
Generate the three required PDF documents for the coursework submission:
  - requirements.pdf
  - manual.pdf
  - replication.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT


def make_code_style(styles):
    return ParagraphStyle(
        'Code', parent=styles['Normal'],
        fontName='Courier', fontSize=9, leading=12,
        leftIndent=12, spaceAfter=8
    )


def build_requirements_pdf():
    doc = SimpleDocTemplate("docs/requirements.pdf", pagesize=A4,
                            topMargin=25*mm, bottomMargin=25*mm,
                            leftMargin=25*mm, rightMargin=25*mm)
    styles = getSampleStyleSheet()
    code = make_code_style(styles)
    story = []

    story.append(Paragraph("Requirements", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("System Requirements", styles['Heading2']))
    story.append(Paragraph("Python 3.8 or later is required. The tool has been tested on Python 3.10 and 3.11.", styles['Normal']))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Python Package Dependencies", styles['Heading2']))
    deps = [
        ("tensorflow", ">=2.0", "Pre-trained model loading and inference"),
        ("numpy", ">=1.21", "Numerical operations and array manipulation"),
        ("pandas", ">=1.3", "Data loading and DataFrame operations"),
        ("scikit-learn", ">=1.0", "Train/test splitting"),
        ("scipy", ">=1.7", "Wilcoxon signed-rank test and statistical analysis"),
        ("matplotlib", ">=3.5", "Figure generation for the report"),
    ]
    for pkg, ver, desc in deps:
        story.append(Paragraph(
            f"<b>{pkg}</b> {ver} -- {desc}", styles['Normal']
        ))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Installation", styles['Heading2']))
    story.append(Preformatted("pip install -r requirements.txt", code))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Dataset and Model Files", styles['Heading2']))
    story.append(Paragraph(
        "The pre-trained models (.h5) and datasets (.csv) must be downloaded from the ISE Lab 4 "
        "repository: https://github.com/ideas-labo/ISE/tree/main/lab4 and placed in the model/ "
        "directory at the project root.", styles['Normal']
    ))

    doc.build(story)
    print("Created docs/requirements.pdf")


def build_manual_pdf():
    doc = SimpleDocTemplate("docs/manual.pdf", pagesize=A4,
                            topMargin=25*mm, bottomMargin=25*mm,
                            leftMargin=25*mm, rightMargin=25*mm)
    styles = getSampleStyleSheet()
    code = make_code_style(styles)
    story = []

    story.append(Paragraph("User Manual", styles['Title']))
    story.append(Paragraph("Two-Phase Directed Search for AI Model Fairness Testing", styles['Heading3']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Overview", styles['Heading2']))
    story.append(Paragraph(
        "This tool tests pre-trained AI models for individual fairness by generating input pairs "
        "that differ only on sensitive features and checking whether the model produces different "
        "predictions. It implements a Two-Phase Directed Search that improves upon the Random Search "
        "baseline by exploiting the clustering property of discriminatory regions.",
        styles['Normal']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Setup", styles['Heading2']))
    story.append(Paragraph("1. Install dependencies:", styles['Normal']))
    story.append(Preformatted("pip install -r requirements.txt", code))
    story.append(Paragraph(
        "2. Download datasets and models from the ISE Lab 4 GitHub repository and place them "
        "in the model/ directory.", styles['Normal']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Running Experiments", styles['Heading2']))
    story.append(Paragraph("Navigate to the src/ directory and run:", styles['Normal']))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Run on specific datasets:", styles['Normal']))
    story.append(Preformatted("python main.py --datasets kdd adult compas --runs 30", code))

    story.append(Paragraph("Run on all available datasets:", styles['Normal']))
    story.append(Preformatted("python main.py --all --runs 30", code))

    story.append(Paragraph("Run with custom budget:", styles['Normal']))
    story.append(Preformatted("python main.py --datasets kdd --runs 30 --budget 2000", code))

    story.append(Paragraph("Generate figures from existing results:", styles['Normal']))
    story.append(Preformatted("python main.py --visualize", code))

    story.append(Paragraph("List available datasets:", styles['Normal']))
    story.append(Preformatted("python main.py", code))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Outputs", styles['Heading2']))
    story.append(Paragraph(
        "Results are saved in results/raw_results.csv and results/summary_statistics.csv. "
        "Figures are saved in the figures/ directory as PNG files at 300 DPI, ready for inclusion "
        "in the report.", styles['Normal']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Configuration", styles['Heading2']))
    story.append(Paragraph(
        "Experiment parameters can be modified in src/config.py. Key parameters include the "
        "budget (max unique inputs), number of runs, discrimination threshold, local search "
        "radius, local search steps per seed, and the global/local budget split fraction.",
        styles['Normal']
    ))

    doc.build(story)
    print("Created docs/manual.pdf")


def build_replication_pdf():
    doc = SimpleDocTemplate("docs/replication.pdf", pagesize=A4,
                            topMargin=25*mm, bottomMargin=25*mm,
                            leftMargin=25*mm, rightMargin=25*mm)
    styles = getSampleStyleSheet()
    code = make_code_style(styles)
    story = []

    story.append(Paragraph("Replication Instructions", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Step 1: Clone the Repository", styles['Heading2']))
    story.append(Preformatted("git clone https://github.com/shaantshor/fairness-testing.git\ncd fairness-testing", code))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Step 2: Install Dependencies", styles['Heading2']))
    story.append(Preformatted("pip install -r requirements.txt", code))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Step 3: Download Datasets and Models", styles['Heading2']))
    story.append(Paragraph(
        "Download the datasets (.csv) and pre-trained models (.h5) from the ISE Lab 4 "
        "repository at https://github.com/ideas-labo/ISE/tree/main/lab4. "
        "Place all files in the model/ directory.",
        styles['Normal']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Step 4: Run the Full Experiment", styles['Heading2']))
    story.append(Paragraph(
        "To replicate all results reported in the paper, run the following command "
        "from the src/ directory:", styles['Normal']
    ))
    story.append(Preformatted("cd src\npython main.py --all --runs 30", code))
    story.append(Paragraph(
        "This runs both the Random Search baseline and the Two-Phase Directed Search "
        "on all 8 datasets, across all sensitive features, for 30 independent runs each. "
        "The total runtime depends on hardware but typically takes 2-4 hours.",
        styles['Normal']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Step 5: Verify Results", styles['Heading2']))
    story.append(Paragraph(
        "After completion, check results/summary_statistics.csv for the IDI ratios, Wilcoxon "
        "p-values, and effect sizes. Figures will be generated automatically in the figures/ "
        "directory. Due to the stochastic nature of the algorithms, exact numerical values may "
        "differ slightly from those reported, but the relative improvement and statistical "
        "significance should be consistent.",
        styles['Normal']
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Step 6: Reproduce Specific Results", styles['Heading2']))
    story.append(Paragraph(
        "To reproduce results for a specific dataset:", styles['Normal']
    ))
    story.append(Preformatted("python main.py --datasets kdd --runs 30", code))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Configuration Used in the Report", styles['Heading2']))
    story.append(Paragraph("The experiments use the following parameters:", styles['Normal']))
    params = [
        "Budget: 1000 unique inputs per run",
        "Runs: 30 per dataset/sensitive-feature combination",
        "Threshold: 0.05 (prediction difference for discrimination)",
        "Global search fraction: 0.3 (30% of budget for Phase 1)",
        "Local search radius: 0.1 (fraction of feature range)",
        "Local search steps: 10 perturbations per seed",
        "Random state: 42 (base seed, each run adds its index)",
    ]
    for p in params:
        story.append(Paragraph(f"  - {p}", styles['Normal']))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Raw Data", styles['Heading2']))
    story.append(Paragraph(
        "The raw results from all 30 runs are included in results/raw_results.csv in the "
        "repository. This file contains per-run IDI ratios, discriminatory instance counts, "
        "and execution times for both approaches.",
        styles['Normal']
    ))

    doc.build(story)
    print("Created docs/replication.pdf")


if __name__ == "__main__":
    import os
    os.makedirs("docs", exist_ok=True)
    build_requirements_pdf()
    build_manual_pdf()
    build_replication_pdf()
