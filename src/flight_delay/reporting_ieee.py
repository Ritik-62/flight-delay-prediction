"""Generator for IEEE two-column conference paper in both LaTeX (.tex) and compiled PDF formats."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)


class NumberedCanvasIEEE(canvas.Canvas):
    """IEEE conference publication canvas with clean page numbering and neutral running headers."""

    def __init__(self, *args, **kwargs):
        kwargs["pageCompression"] = 0
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor("#333333"))

        # Neutral Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                0.5 * inch,
                10.5 * inch,
                "Applied Machine Learning --- Phase 2 Technical Report --- November 2026",
            )
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(0.5 * inch, 10.42 * inch, 8.0 * inch, 10.42 * inch)

        # Neutral Footer with standard page numbering
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.0 * inch, 0.35 * inch, footer_text)
        self.restoreState()


class IEEEConferenceReportGenerator:
    """Produces authentic IEEE two-column camera-ready conference papers (.tex and .pdf)."""

    def __init__(self, reports_dir: Path, figures_dir: Path):
        self.reports_dir = Path(reports_dir)
        self.figures_dir = Path(figures_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_latex_document(
        self,
        metrics_df: pd.DataFrame,
        rescheduling_summary: pd.DataFrame,
        output_tex_path: Path,
    ) -> Path:
        """Write camera-ready IEEEtran two-column LaTeX file."""
        output_tex_path = Path(output_tex_path)

        table_rows = []
        for _, row in metrics_df.iterrows():
            m_name = str(row["Model"]).replace("_", "\\_").replace("%", "\\%")
            roc = f"\\mbox{{{row.get('ROC AUC', 0):.4f}}}"
            pr = f"\\mbox{{{row.get('PR AUC', 0):.4f}}}"
            p = f"\\mbox{{{row.get('Precision', 0):.4f}}}"
            r = f"\\mbox{{{row.get('Recall', 0):.4f}}}"
            f1 = f"\\mbox{{{row.get('F1 Score', 0):.4f}}}"
            br = f"\\mbox{{{row.get('Brier Score', 0):.5f}}}"
            lat_val = row.get('Latency (ms/1k)')
            lat = f"\\mbox{{{lat_val:.2f}}}" if pd.notna(lat_val) else "--"
            table_rows.append(f"{m_name} & {roc} & {pr} & {p} & {r} & {f1} & {br} & {lat} \\\\")

        metrics_latex_table = "\n".join(table_rows)

        resched_rows = []
        for _, r in rescheduling_summary.iterrows():
            tau = f"\\mbox{{{r.get('Risk Threshold (tau)', 0):.3f}}}"
            ir = f"\\mbox{{{r.get('Intervention Rate (%)', 0):.1f}}}\\%"
            cov = f"\\mbox{{{r.get('Delay Coverage Rate (%)', 0):.1f}}}\\%"
            av = f"\\mbox{{{int(r.get('Estimated Averted Delayed Flights', 0)):,}}}"
            gross_val = r.get('Gross Delay Minutes Saved', 0)
            gross = f"\\mbox{{{gross_val:,.1f}}}"
            net_val = r.get('Net Benefit (Min Saved - Friction)', 0)
            # Highlight negative net benefit in red
            if net_val < 0:
                net = f"\\textcolor{{red}}{{\\mbox{{{net_val:,.1f}}}}}"
            else:
                net = f"\\mbox{{{net_val:,.1f}}}"
            resched_rows.append(f"{tau} & {ir} & {cov} & {av} & {gross} & {net} \\\\")
        resched_latex_table = "\n".join(resched_rows)

        template = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{url}
\usepackage{array}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

\begin{document}

\title{Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning:\\
Pre-Departure Risk Modeling and Tactical Rescheduling\\
{\footnotesize \textsuperscript{*}Phase 2 Technical Report --- Applied Machine Learning Project}
}

\author{\IEEEauthorblockN{Vaibhav, Parv, Ayush, Ritik, Mohit}
\IEEEauthorblockA{Applied Machine Learning Project Team\\
Course Project Deliverable}
}

\maketitle

\begin{abstract}
Predicting commercial flight arrival delays prior to gate departure poses formidable statistical challenges: models must strictly avoid post-departure target leakage while capturing complex interdependencies among airport congestion, scheduled block buffers, carrier network dynamics, and propagated inbound aircraft turnaround delays. We establish an end-to-end machine learning framework comparing tabular gradient boosting classifiers (XGBoost, LightGBM), linear baselines, and deep neural architectures (Tabular Entity-Embedding MLP and Recurrent Temporal Networks) evaluated on US domestic flights from the Bureau of Transportation Statistics (BTS 2015 dataset, 147,389 flights across 358 calendar dates). Operating strictly at decision epoch $T - 60$ minutes prior to scheduled departure, our feature engineering encapsulates expanding empirical Bayes smoothed historical priors, hourly departure/arrival density, and leakage-safe inbound turnaround slack. Empirical evaluation under calendar-date chronological partitioning confirms that the Deep Tabular MLP achieves the highest discriminative performance (ROC-AUC \mbox{0.6196}, PR-AUC \mbox{0.2319}), outperforming gradient boosted ensembles (XGBoost ROC-AUC \mbox{0.5982}, PR-AUC \mbox{0.2141}; LightGBM \mbox{0.5582}/\mbox{0.1875}) and linear baselines (\mbox{0.6159}/\mbox{0.2284}). Post-hoc Platt sigmoid calibration compresses the Brier score from \mbox{0.18734} to \mbox{0.13271} and reduces the Expected Calibration Error (ECE) from \mbox{0.2255} to \mbox{0.0389}. At threshold $\tau = 0.3637$, the calibrated MLP achieves \mbox{71.43\%} precision and \mbox{0.29\%} recall; we clarify this is a threshold-specific trade-off, not overall superiority. An offline rescheduling simulation on 21,866 test flights at $\tau = 0.171$ intervenes on \mbox{2,905} flights (\mbox{13.29\%}), capturing \mbox{759} delays (\mbox{21.69\%} coverage) with \mbox{2,146} false alarms. Under simulation assumptions: \mbox{303} estimated averted delays, \mbox{9,176.7} gross delay minutes saved. After schedule friction (\mbox{7.5} penalty min/intervention, \mbox{21,787.5} total), net benefit is \textcolor{red}{\mbox{$-$12,610.8} minutes}---negative across all eight thresholds.
\end{abstract}

\begin{IEEEkeywords}
Flight delay prediction, gradient boosting, temporal deep learning, entity embeddings, probability calibration, data leakage, airline rescheduling.
\end{IEEEkeywords}

\section{Introduction}
Commercial aviation networks operate as tightly coupled, high-utilization queueing systems. A minor disturbance at a hub airport frequently cascades across the national airspace system, generating severe downstream delay propagation. In response, airline dispatchers and air traffic control command centers require predictive decision support capable of forecasting arrival delays of 15 minutes or greater (\texttt{ARR\_DEL15}) well before departure.
However, standard machine learning pipelines frequently suffer from severe data leakage by conditioning on actual departure delays, ground taxi-out times, or post-departure telemetry. In this work, we enforce a strict operational boundary: \textbf{all features must be definitively known 60 minutes prior to scheduled departure ($T - 60$ min)}. We benchmark linear baselines, gradient boosted decision trees (GBDTs), tabular neural networks with learned entity embeddings, and temporal sequential models, followed by an offline tactical rescheduling simulation.

\section{Data Leakage Prevention \& Chronological Partitioning}
\label{sec:data}
Operational fidelity demands a mathematically rigorous audit preventing target or post-decision contamination.
\subsection{The Denylist Protocol}
We enforce an automated audit denying any column reflecting events occurring after $T - 60$ minutes. Excluded variables include \texttt{DEP\_DELAY}, \texttt{DEPARTURE\_TIME}, \texttt{TAXI\_OUT}, \texttt{WHEELS\_OFF}, \texttt{AIR\_TIME}, and causal delay attributes (e.g., \texttt{WEATHER\_DELAY}, \texttt{LATE\_AIRCRAFT\_DELAY}).
\subsection{Chronological Date Partitioning}
Random $K$-fold cross-validation inflates benchmark performance due to high autocorrelation across consecutive calendar days. To ensure unbiased generalization, we enforce calendar-date chronological splitting across the 358 operational dates of 2015:
\begin{itemize}
    \item \textbf{Training Set (70.39\%):} 2015-01-01 to 2015-09-07 (250 dates, 103,747 flights).
    \item \textbf{Validation Set (14.77\%):} 2015-09-08 to 2015-10-30 (53 dates, 21,776 flights).
    \item \textbf{Held-out Test Set (14.84\%):} 2015-10-31 to 2015-12-24 (55 dates, 21,866 flights).
\end{itemize}

\subsection{Dataset Provenance \& Cross-Phase Comparability}
Both Phase 1 and Phase 2 derive from the official 2015 U.S.\ DOT BTS Airline On-Time Performance dataset (\texttt{flights.parquet}, 5,819,079 domestic records). Phase 1 sampled 350,000 records (343,885 valid flights; test: 51,647 flights, Oct 25--Dec 17). To support deep sequence models and Optuna hyperparameter sweeps within memory constraints, Phase 2 employed a uniform step-subsample of 150,000 records (147,389 valid flights; test: 21,866 flights, Oct 31--Dec 24). Because sample sizes, date counts, and test periods differ, \textbf{Phase 1 and Phase 2 metrics are not directly comparable across different test populations}. To guarantee strict internal validity, all baselines (Logistic Regression, XGBoost) were retrained and evaluated on the exact same Phase 2 held-out test partition alongside the deep neural models.

\section{Feature Engineering: Congestion \& Turnaround Slack}
Phase 2 extends tabular predictors by modeling operational bottlenecks:
\subsection{Expanding Empirical Bayes Historical Priors}
To prevent target leakage across training dates, daily cumulative delay statistics are computed strictly on past dates ($t < D$). For entity $e$ (airline, origin, destination, route), the smoothed delay rate is:
\begin{equation}
\hat{p}_{e, D} = \frac{\sum_{t < D} \text{Delays}_{e, t} + \alpha \cdot \bar{p}_{\text{global}}}{\sum_{t < D} \text{Flights}_{e, t} + \alpha}
\end{equation}
where $\alpha = 15.0$ acts as a pseudo-count shrinkage parameter.
\subsection{Airport Rolling Congestion}
We quantify terminal traffic density using published flight schedules: $\text{ORIGIN\_CONGESTION\_1H}$ computes scheduled departures within a $\pm 60$-minute departure window; $\text{DEST\_CONGESTION\_1H}$ measures scheduled arrivals in the destination bank.
\subsection{Aircraft Turnaround Delay Propagation}
For each aircraft tail number (\texttt{TAIL\_NUMBER}), we track the inbound flight leg. At decision epoch $T - 60$ min:
\begin{itemize}
    \item If the inbound flight landed prior to $T - 60$, its true delay is observed (\texttt{INBOUND\_DELAY\_KNOWN}).
    \item If the inbound flight has not landed, \texttt{INBOUND\_UNARRIVED\_AT\_T60} is set to 1.
    \item The scheduled turnaround buffer is $\Delta T_{\text{buffer}} = T_{\text{sched\_dep}} - T_{\text{prev\_sched\_arr}}$.
\end{itemize}

\section{Model Architectures \& Deep Learning}
\subsection{Gradient Boosted Ensembles}
We implement XGBoost (histogram tree method) and LightGBM with tree depth regularization and positive class weighting to mitigate class imbalance.
\subsection{Deep Tabular MLP with Entity Embeddings}
High-cardinality categorical variables are mapped into low-dimensional dense embedding vectors:
\begin{equation}
\mathbf{e}_{\text{air}} \in \mathbb{R}^{16}, \quad \mathbf{e}_{\text{orig}} \in \mathbb{R}^{32}, \quad \mathbf{e}_{\text{dest}} \in \mathbb{R}^{32}
\end{equation}
Dense representations are concatenated with standardized continuous variables and passed through a three-layer MLP with Batch Normalization, Mish activations, and Dropout (0.2).
\subsection{Temporal Sequence Recurrent Model}
A 2-layer Gated Recurrent Unit (GRU) models sequential turnaround states across aircraft itinerary legs, capturing temporal delay accumulation.

\section{Probability Calibration}
Modern deep networks and boosted trees often output miscalibrated posterior probabilities. We apply Platt (sigmoid) calibration fitted strictly on validation logits:
\begin{equation}
\hat{P}(Y=1 \mid z) = \sigma(A z + B)
\end{equation}
Calibration performance is evaluated using the Brier score and Expected Calibration Error (ECE) across $M=10$ uniform bins:
\begin{equation}
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
\end{equation}
Empirically, Platt calibration on the held-out test set reduced the Brier score from \mbox{0.18734} (raw MLP) to \mbox{0.13271} and compressed the ECE from \mbox{0.2255} to \mbox{0.0389}, producing dependable probability estimates for downstream tactical dispatch.

\section{Experimental Results \& Benchmark Comparison}
Table~\ref{tab:benchmark} presents comprehensive out-of-sample performance on the held-out test partition (21,866 flights, Oct~31--Dec~24, 2015). All baselines were retrained on the identical Phase~2 test partition to ensure strict comparability.

\begin{table*}[!t]
\renewcommand{\arraystretch}{1.2}
\caption{Model Benchmark Comparison on Held-Out Test Partition ($T - 60$~min Pre-Departure, 21,866 Flights).
  ``Cal.'' = Platt-calibrated MLP at high-precision threshold $\tau = 0.3637$; LightGBM Prec/Rec/F1 $= 0$ at default $\tau=0.5$ due to class imbalance---use PR-AUC for ranking.}
\label{tab:benchmark}
\centering
\setlength{\tabcolsep}{7pt}
\begin{tabular}{@{}l r r r r r r r@{}}
\toprule
\textbf{Model} &
  \textbf{ROC-AUC} &
  \textbf{PR-AUC} &
  \textbf{Precision} &
  \textbf{Recall} &
  \textbf{F1} &
  \textbf{Brier\,$\downarrow$} &
  \textbf{Lat.\,(ms/1k)} \\
\midrule
__METRICS_TABLE__
\bottomrule
\end{tabular}
\end{table*}

The Deep Tabular MLP with entity embeddings achieved the highest discriminative performance (ROC-AUC \mbox{0.6196}, PR-AUC \mbox{0.2319}), demonstrating that learned categorical representations capture complex non-linear hub and route interactions more effectively than gradient boosted trees (XGBoost ROC-AUC \mbox{0.5982}, PR-AUC \mbox{0.2141}; LightGBM ROC-AUC \mbox{0.5582}, PR-AUC \mbox{0.1875}) and linear baselines (ROC-AUC \mbox{0.6159}, PR-AUC \mbox{0.2284}).

\subsection{Operational Precision and Threshold-Specific Behavior}
When tuned to meet the operational precision constraint ($\ge 70\%$) on the validation partition, the calibrated Tabular MLP identified an operating threshold of $\tau = 0.3637$. On the held-out test set, this yields a precision of \mbox{0.7143} (\mbox{71.43\%}) and a recall of \mbox{0.0029} (\mbox{0.29\%}). We stress that this high precision is an operating-point artifact resulting from an intentionally conservative threshold that triggers on only \mbox{0.3\%} of delayed flights. It must not be conflated with overall discriminative superiority, which is properly characterized by PR-AUC (\mbox{0.2319}) and ROC-AUC (\mbox{0.6196}).

\section{Offline Tactical Rescheduling Analysis}
\subsection{Operational Policy Formulation}
When the calibrated delay risk exceeds policy threshold $\tau$, an offline recommendation is triggered:
\begin{itemize}
    \item \textbf{Tail Swap:} If the aircraft has a compromised turnaround buffer at a major carrier hub (share $\ge 25\%$), swap with an available on-time aircraft.
    \item \textbf{Slot Shift (+30 min):} For non-hub or congestion-driven risk, insert a 30-minute ground delay buffer to absorb downstream queueing.
\end{itemize}

\subsection{Policy Sweep \& Intervention Friction Analysis}
Table~\ref{tab:resched} summarizes the counterfactual simulation across eight risk thresholds on 21,866 test flights. \textbf{Net benefit (column 6) is negative at every threshold}, shown in \textcolor{red}{red}, reflecting that schedule friction exceeds gross delay savings under this linear penalty model.

\begin{table}[!ht]
\renewcommand{\arraystretch}{1.1}
\caption{Rescheduling Policy Sweep (21,866 Test Flights). Net\,Min = Gross Delay Minutes $-$ Friction Penalty (7.5\,min $\times$ interventions). \textcolor{red}{All net values are negative} under this linear cost model.}
\label{tab:resched}
\centering
\setlength{\tabcolsep}{3.5pt}
\small
\begin{tabular}{@{}r r r r r r@{}}
\toprule
$\boldsymbol{\tau}$ &
  \textbf{Interv.\,\%} &
  \textbf{Cov.\,\%} &
  \textbf{Averted} &
  \textbf{Gross\,Min} &
  \textbf{Net\,Min} \\
\midrule
__RESCHED_TABLE__
\bottomrule
\end{tabular}
\end{table}

At operational threshold $\tau = 0.171$, the policy intervenes on \mbox{2,905} flights (\mbox{13.29\%}), capturing \mbox{759} actual delays (\mbox{21.69\%} delay coverage) while generating \mbox{2,146} false alarm interventions. Under the simulation model assumptions, this yields an estimated \mbox{303} averted delays and \mbox{9,176.7} gross delay minutes saved (simulation metric based on 45\,min average delay duration; not passenger-level delay minutes).

Crucially, the net benefit is \textcolor{red}{negative (\mbox{$-$12,610.8} minutes)}. Under the simulation friction model, each intervention incurs a schedule buffer penalty of $30\,\text{min} \times 0.25 = 7.5\,\text{penalty minutes}$, totaling \mbox{21,787.5} penalty minutes. Because false alarm interventions (\mbox{2,146} flights, \mbox{73.87\%} of interventions) bear this penalty without averting an actual delay, the friction cost exceeds gross savings. Net benefit remains negative across all eight evaluated thresholds under this linear cost formulation.

\section{Conclusion}
Phase~2 successfully unified leakage-safe operational feature engineering with gradient boosting, tabular neural embeddings, and temporal deep learning. The Deep Tabular MLP with entity embeddings attained the highest discriminative performance (ROC-AUC \mbox{0.6196}, PR-AUC \mbox{0.2319}), while Platt calibration reduced ECE from \mbox{0.2255} to \mbox{0.0389}. Tactical rescheduling counterfactuals demonstrate that proactive interventions impose substantial schedule friction, resulting in negative net benefit under linear penalty models---underscoring the need for accurate delay probability estimation before operational deployment.

\begin{thebibliography}{00}
\bibitem{bts} U.S. Bureau of Transportation Statistics, ``Airline On-Time Performance Data (2015),'' Bureau of Transportation Statistics, U.S. Department of Transportation, 2015. [Online]. Available: https://www.bts.gov/
\bibitem{guo2017} C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, ``On Calibration of Modern Neural Networks,'' in \textit{Proc. 34th Int. Conf. Machine Learning (ICML)}, 2017, pp. 1321--1330.
\bibitem{chen2016} T. Chen and C. Guestrin, ``XGBoost: A Scalable Tree Boosting System,'' in \textit{Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining}, 2016, pp. 785--794.
\bibitem{ke2017} G. Ke et al., ``LightGBM: A Highly Efficient Gradient Boosting Decision Tree,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 2017, pp. 3146--3154.
\bibitem{guo2016} C. Guo and F. Berkhahn, ``Entity Embeddings of Categorical Variables,'' \textit{arXiv:1604.06737}, 2016.
\end{thebibliography}

\end{document}
"""
        tex_content = template.replace("__METRICS_TABLE__", metrics_latex_table).replace("__RESCHED_TABLE__", resched_latex_table)
        with open(output_tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)

        logger.info(f"Wrote IEEE LaTeX paper to {output_tex_path}.")
        return output_tex_path

    def generate_pdf_document(
        self,
        metrics_df: pd.DataFrame,
        rescheduling_summary: pd.DataFrame,
        figures_paths: Dict[str, Path],
        output_pdf_path: Path,
    ) -> Path:
        """Render IEEE two-column formatted camera-ready PDF document."""
        output_pdf_path = Path(output_pdf_path)

        page_width, page_height = letter
        margin = 0.5 * inch
        col_width = 3.6 * inch
        gutter = 0.3 * inch

        doc = BaseDocTemplate(
            str(output_pdf_path),
            pagesize=letter,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
        )

        frame_title = Frame(margin, page_height - margin - 1.5 * inch, 7.5 * inch, 1.5 * inch, id="F_Title", topPadding=0, bottomPadding=0)
        frame_p1_c1 = Frame(margin, margin, col_width, page_height - margin - 1.65 * inch, id="F_P1_C1", topPadding=0, bottomPadding=0)
        frame_p1_c2 = Frame(margin + col_width + gutter, margin, col_width, page_height - margin - 1.65 * inch, id="F_P1_C2", topPadding=0, bottomPadding=0)

        frame_col1 = Frame(margin, margin, col_width, page_height - 2 * margin - 0.1 * inch, id="F_Col1", topPadding=5, bottomPadding=5)
        frame_col2 = Frame(margin + col_width + gutter, margin, col_width, page_height - 2 * margin - 0.1 * inch, id="F_Col2", topPadding=5, bottomPadding=5)

        p1_template = PageTemplate(id="Page1", frames=[frame_title, frame_p1_c1, frame_p1_c2])
        two_col_template = PageTemplate(id="TwoCol", frames=[frame_col1, frame_col2])

        doc.addPageTemplates([p1_template, two_col_template])

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "IEEETitle",
            fontName="Times-Bold",
            fontSize=15,
            leading=18,
            alignment=1,
            textColor=colors.HexColor("#111111"),
        )
        subtitle_style = ParagraphStyle(
            "IEEESubTitle",
            fontName="Times-Italic",
            fontSize=9.5,
            leading=12,
            alignment=1,
            textColor=colors.HexColor("#333333"),
        )
        author_style = ParagraphStyle(
            "IEEEAuthor",
            fontName="Times-Roman",
            fontSize=8.5,
            leading=11.5,
            alignment=1,
            textColor=colors.HexColor("#222222"),
        )
        h1_style = ParagraphStyle(
            "IEEEHeading1",
            fontName="Times-Bold",
            fontSize=8.8,
            leading=10.8,
            spaceBefore=5,
            spaceAfter=2,
            textColor=colors.HexColor("#000000"),
        )
        body_style = ParagraphStyle(
            "IEEEBody",
            fontName="Times-Roman",
            fontSize=7.6,
            leading=9.6,
            alignment=4,
            spaceAfter=2,
            textColor=colors.HexColor("#111111"),
        )
        abstract_style = ParagraphStyle(
            "IEEEAbstract",
            fontName="Times-BoldItalic",
            fontSize=7.2,
            leading=9.0,
            alignment=4,
            leftIndent=4,
            rightIndent=4,
            spaceAfter=3,
        )
        table_cell_style = ParagraphStyle(
            "IEEETableCell",
            fontName="Times-Roman",
            fontSize=6.5,
            leading=8,
            alignment=1,
        )
        table_header_style = ParagraphStyle(
            "IEEETableHeader",
            fontName="Times-Bold",
            fontSize=7,
            leading=8.5,
            alignment=1,
            textColor=colors.whitesmoke,
        )

        story = []

        # Title Frame
        story.append(Paragraph("Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning: Pre-Departure Risk Modeling and Tactical Rescheduling", title_style))
        story.append(Spacer(1, 2))
        story.append(Paragraph("Phase 2 Technical Report --- Applied Machine Learning Project", subtitle_style))
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Vaibhav &nbsp;|&nbsp; Parv &nbsp;|&nbsp; Ayush &nbsp;|&nbsp; Ritik &nbsp;|&nbsp; Mohit</b><br/>Applied Machine Learning Project Team --- Course Project Deliverable", author_style))
        story.append(Spacer(1, 3))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#333333"), spaceBefore=2, spaceAfter=2))
        story.append(FrameBreak())

        # Page 1 Column 1
        story.append(Paragraph("<b><i>Abstract</i>—Predicting commercial flight arrival delays prior to gate departure poses formidable statistical challenges: models must strictly avoid post-departure target leakage while capturing complex interdependencies among airport congestion, scheduled block buffers, carrier network dynamics, and propagated inbound aircraft turnaround delays. In this paper, we establish an end-to-end framework comparing gradient boosted decision trees (XGBoost, LightGBM), linear baselines, and deep neural architectures (Tabular Entity-Embedding MLP and Recurrent Temporal Networks) evaluated on commercial US domestic flights from the Bureau of Transportation Statistics (BTS 2015 dataset, 147,389 flights across 358 calendar dates). Operating strictly at decision epoch T - 60 minutes prior to scheduled departure, our feature engineering encapsulates expanding empirical Bayes smoothed historical priors, hourly departure/arrival density, and leakage-safe inbound turnaround slack. Empirical evaluation under calendar-date chronological partitioning confirms that the Deep Tabular MLP with dense entity embeddings achieves the highest discriminative performance (ROC AUC 0.6196, PR AUC 0.2319), outperforming gradient boosted ensembles (XGBoost ROC AUC 0.5982, PR AUC 0.2141; LightGBM ROC AUC 0.5582, PR AUC 0.1875) and linear baselines (ROC AUC 0.6159, PR AUC 0.2284). Post-hoc Platt sigmoid calibration compresses the Brier score from 0.18734 to 0.13271 and reduces the Expected Calibration Error (ECE) from 0.2255 to 0.0389. Operating at decision threshold tau = 0.3637, the calibrated MLP satisfies the operational precision constraint (&ge; 70%, achieving 71.43% precision and 0.29% recall, capturing 0.3% of delayed flights); we clarify that this represents a threshold-specific trade-off to minimize false alarms rather than overall model superiority. Finally, we formulate an offline tactical rescheduling simulation based on explicit operational assumptions. Evaluated on 21,866 test flights at risk threshold tau = 0.171, the policy intervenes on 2,905 flights (13.29% intervention rate), capturing 759 actual delays (21.69% delay coverage) with 2,146 false alarm interventions. Under simulation assumptions, this yields an estimated 303 averted delays and 9,176.7 gross delay minutes saved. However, factoring in schedule friction (7.5 penalty minutes per intervention, totaling 21,787.5 penalty minutes) results in a negative net benefit of -12,610.8 minutes, demonstrating that proactive interventions impose substantial operational friction that exceeds gross savings under linear cost models.</b>", abstract_style))
        story.append(Paragraph("<b><i>Keywords</i>—Flight delay prediction, gradient boosting, temporal deep learning, entity embeddings, probability calibration, data leakage, airline rescheduling.</b>", body_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#999999"), spaceBefore=2, spaceAfter=3))

        story.append(Paragraph("I. INTRODUCTION", h1_style))
        story.append(Paragraph("Commercial aviation networks operate as tightly coupled queueing systems. A minor disturbance at a hub airport cascades across the national airspace system, generating downstream delay propagation. In response, airline dispatchers and air traffic control command centers require predictive decision support capable of forecasting arrival delays of 15 minutes or greater (<code>ARR_DEL15</code>) well before departure.", body_style))
        story.append(Paragraph("However, standard machine learning pipelines frequently suffer from severe data leakage by conditioning on actual departure delays, ground taxi-out times, or post-departure telemetry. In this work, we enforce a strict operational boundary: <b>all features must be definitively known 60 minutes prior to scheduled departure (T - 60 min)</b>. We benchmark linear baselines, gradient boosted decision trees (GBDTs), tabular neural networks with learned entity embeddings, and temporal sequential models, followed by an offline tactical rescheduling simulation.", body_style))

        story.append(Paragraph("II. DATA INTEGRITY & LEAKAGE PREVENTION", h1_style))
        story.append(Paragraph("Operational fidelity demands a mathematically rigorous audit preventing target or post-decision contamination. We enforce an automated audit denying any column reflecting events occurring after T - 60 minutes. Excluded variables include <code>DEP_DELAY</code>, <code>DEPARTURE_TIME</code>, <code>TAXI_OUT</code>, <code>WHEELS_OFF</code>, <code>AIR_TIME</code>, and causal delay attributes (e.g., <code>WEATHER_DELAY</code>, <code>LATE_AIRCRAFT_DELAY</code>).", body_style))
        story.append(Paragraph("Random K-fold cross-validation inflates benchmark performance due to high autocorrelation across consecutive calendar days. To guarantee zero leakage, we enforce calendar-date chronological splitting across 2015: Training (2015-01-01 to 2015-09-07, 103,747 flights), Validation (2015-09-08 to 2015-10-30, 21,776 flights), and Test (2015-10-31 to 2015-12-24, 21,866 flights).", body_style))
        story.append(Paragraph("<b>Cross-Phase Comparability:</b> Phase 1 sampled 350,000 records (test: 51,647 flights, Oct 25–Dec 17), while Phase 2 sampled 150,000 records (test: 21,866 flights, Oct 31–Dec 24) to accommodate deep neural sequence architectures and Optuna tuning within memory budgets. Because test populations differ, <b>Phase 1 and Phase 2 metrics are not directly comparable</b>; all baselines in Table I were re-evaluated on the identical Phase 2 test set.", body_style))

        # Page 1 Column 2 flows naturally
        story.append(Paragraph("III. ADVANCED FEATURE ENGINEERING", h1_style))
        story.append(Paragraph("<b>A. Expanding Empirical Bayes Historical Priors:</b> To prevent target leakage across training dates, daily cumulative delay statistics are computed strictly on past dates (t < D). For entity e (airline, origin, destination, route), the smoothed delay rate shrinks low-volume categories toward the global prior using smoothing parameter alpha = 15.0.", body_style))
        story.append(Paragraph("<b>B. Airport Rolling Congestion:</b> Terminal traffic density is quantified using published flight schedules: <code>ORIGIN_CONGESTION_1H</code> computes scheduled departures within a ±60-minute departure window; <code>DEST_CONGESTION_1H</code> measures scheduled arrivals in the destination bank.", body_style))
        story.append(Paragraph("<b>C. Aircraft Turnaround Delay Propagation:</b> For each aircraft tail number (<code>TAIL_NUMBER</code>), we track the inbound flight leg. At decision epoch T - 60 min: if the inbound flight landed prior to T - 60, its true delay is observed (<code>INBOUND_DELAY_KNOWN</code>); if unlanded, <code>INBOUND_UNARRIVED_AT_T60</code> is flagged.", body_style))

        story.append(Paragraph("IV. MODEL ARCHITECTURES", h1_style))
        story.append(Paragraph("<b>A. Tuned Gradient Boosters:</b> We implement XGBoost and LightGBM with tree depth regularization, histogram-based binning, and positive class weighting to mitigate class imbalance.", body_style))
        story.append(Paragraph("<b>B. Deep Tabular MLP with Entity Embeddings:</b> High-cardinality categorical variables are mapped into dense embedding vectors (Airline: 16d, Origin: 32d, Destination: 32d) concatenated with continuous predictors and passed through a three-layer MLP with Batch Normalization, Mish activations, and Dropout (0.2).", body_style))
        story.append(Paragraph("<b>C. Temporal Sequence Recurrent Model:</b> A 2-layer Gated Recurrent Unit (GRU) models sequential turnaround states across aircraft itinerary legs, capturing temporal delay accumulation.", body_style))

        story.append(Paragraph("V. PROBABILITY CALIBRATION", h1_style))
        story.append(Paragraph("Modern deep networks and boosted trees often output miscalibrated posterior probabilities. We apply Platt (sigmoid) scaling fitted strictly on validation logits. Evaluated on the held-out test partition across 10 uniform reliability bins, Platt calibration compressed the Brier score from 0.18734 to 0.13271 and reduced Expected Calibration Error (ECE) from 0.2255 to 0.0389, providing reliable risk probabilities for operational dispatch.", body_style))

        story.append(NextPageTemplate("TwoCol"))
        story.append(PageBreak())

        # Page 2 Column 1
        story.append(Paragraph("VI. EXPERIMENTAL RESULTS &amp; BENCHMARKS", h1_style))
        story.append(Paragraph(
            "Table I reports out-of-sample performance across all models on the held-out test partition "
            "(21,866 flights, Oct 31–Dec 24, 2015, T−60 minutes). "
            "All baselines were retrained on the identical Phase 2 test partition for strict comparability.",
            body_style))

        # --- Model benchmark table: wider columns to prevent numeric wrapping ---
        # Total col_w must fit in col_width (3.6 inch). We use 7 columns.
        # Model name gets 1.18", each of 6 numeric cols gets 0.407" = 3.622" total (fits with rounding)
        num_col_w = 0.395 * inch
        col_w = [1.14 * inch, num_col_w, num_col_w, num_col_w, num_col_w, num_col_w, 0.42 * inch]

        num_style = ParagraphStyle(
            "IEEENum",
            fontName="Times-Roman",
            fontSize=6.5,
            leading=8,
            alignment=2,  # right-align numerics
        )
        bold_num_style = ParagraphStyle(
            "IEEENumBold",
            fontName="Times-Bold",
            fontSize=6.5,
            leading=8,
            alignment=2,
            textColor=colors.HexColor("#003a7a"),
        )

        # Determine best values per numeric column for bold highlighting
        import numpy as np
        roc_vals = metrics_df["ROC AUC"].values
        pr_vals = metrics_df["PR AUC"].values
        best_roc = max(roc_vals)
        best_pr = max(pr_vals)

        table_data = [[
            Paragraph("<b>Model</b>", table_header_style),
            Paragraph("<b>ROC-AUC</b>", table_header_style),
            Paragraph("<b>PR-AUC</b>", table_header_style),
            Paragraph("<b>Prec.</b>", table_header_style),
            Paragraph("<b>Recall</b>", table_header_style),
            Paragraph("<b>F1</b>", table_header_style),
            Paragraph("<b>Brier↓</b>", table_header_style),
        ]]
        for _, row in metrics_df.iterrows():
            m_short = (str(row["Model"])
                       .replace("Logistic Regression", "Log. Reg.")
                       .replace("Calibrated", "Cal.")
                       .replace("DL Tabular MLP", "Tab. MLP")
                       .replace("Temporal Recurrent Net", "Temp. RNN"))
            roc_v = row.get("ROC AUC", 0)
            pr_v = row.get("PR AUC", 0)
            roc_s = bold_num_style if abs(roc_v - best_roc) < 1e-6 else num_style
            pr_s  = bold_num_style if abs(pr_v  - best_pr)  < 1e-6 else num_style
            table_data.append([
                Paragraph(m_short, table_cell_style),
                Paragraph(f"{roc_v:.4f}", roc_s),
                Paragraph(f"{pr_v:.4f}", pr_s),
                Paragraph(f"{row.get('Precision', 0):.4f}", num_style),
                Paragraph(f"{row.get('Recall', 0):.4f}", num_style),
                Paragraph(f"{row.get('F1 Score', 0):.4f}", num_style),
                Paragraph(f"{row.get('Brier Score', 0):.5f}", num_style),
            ])

        t = Table(table_data, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t)
        story.append(Spacer(1, 3))

        story.append(Paragraph(
            "<b>Table I caption:</b> Best discriminator = Tab. MLP (ROC-AUC 0.6196, PR-AUC 0.2319, shown in blue-bold). "
            "LightGBM Prec/Recall/F1 = 0 at default τ=0.5 due to class imbalance; evaluate by PR-AUC. "
            "Cal. MLP at τ=0.3637: Precision 0.7143 / Recall 0.0029 — a threshold-specific trade-off, "
            "not overall superiority.", body_style))
        story.append(Spacer(1, 3))

        if "roc_curves" in figures_paths and figures_paths["roc_curves"].exists():
            story.append(Image(str(figures_paths["roc_curves"]), width=3.5 * inch, height=1.75 * inch))
            story.append(Paragraph(
                "<b>Fig. 1.</b> Out-of-sample ROC curves (held-out test, Oct 31–Dec 24, 2015). "
                "Tab. MLP achieves ROC-AUC 0.6196 (highest). AUC values in legend are area under curve "
                "at the full threshold sweep, not at any single operating point.",
                body_style))
            story.append(Spacer(1, 2))

        if "calibration_curve" in figures_paths and figures_paths["calibration_curve"].exists():
            story.append(Image(str(figures_paths["calibration_curve"]), width=3.5 * inch, height=1.75 * inch))
            story.append(Paragraph(
                "<b>Fig. 2.</b> Reliability diagram (10-bin) before and after Platt sigmoid calibration. "
                "ECE improves from 0.2255 (raw MLP) to 0.0389 (calibrated). "
                "Brier score: 0.18734 → 0.13271. Points near the diagonal indicate well-calibrated probabilities.",
                body_style))
            story.append(Spacer(1, 2))

        story.append(FrameBreak())

        # Page 2 Column 2
        story.append(Paragraph("VII. OFFLINE TACTICAL RESCHEDULING", h1_style))
        story.append(Paragraph(
            "When calibrated delay risk exceeds threshold τ, tactical actions are simulated: "
            "<b>Tail Swap</b> (hub turnarounds) or <b>Slot Shift (+30 min)</b>. "
            "Table II summarises the policy sweep across 21,866 test flights.",
            body_style))
        story.append(Paragraph(
            "<b>At τ = 0.171:</b> 2,905 interventions (13.29%), 759 captured delays (21.69% coverage), "
            "2,146 false alarms, 303 estimated averted delays (simulation), "
            "9,176.7 gross delay minutes saved. "
            "<font color='#cc0000'><b>Net benefit: −2,610.8 min (negative)</b></font>. "
            "Each intervention incurs 7.5 penalty minutes (30 min buffer × 0.25); "
            "2,146 false alarms pay the penalty without averting a delay. "
            "Net benefit is negative at all eight thresholds.",
            body_style))
        story.append(Spacer(1, 3))

        # --- Rescheduling table with red negative net benefit ---
        red_num_style = ParagraphStyle(
            "IEEERedNum",
            fontName="Times-Bold",
            fontSize=6.5,
            leading=8,
            alignment=2,
            textColor=colors.HexColor("#cc0000"),
        )

        resched_data = [[
            Paragraph("<b>τ</b>", table_header_style),
            Paragraph("<b>Interv %</b>", table_header_style),
            Paragraph("<b>Cov %</b>", table_header_style),
            Paragraph("<b>Avrtd</b>", table_header_style),
            Paragraph("<b>Gross Min</b>", table_header_style),
            Paragraph("<b>Net Min</b>", table_header_style),
        ]]
        for _, r in rescheduling_summary.iterrows():
            net_val = r.get("Net Benefit (Min Saved - Friction)", 0)
            net_style = red_num_style  # always red (all negative)
            resched_data.append([
                Paragraph(f"{r.get('Risk Threshold (tau)', 0):.3f}", num_style),
                Paragraph(f"{r.get('Intervention Rate (%)', 0):.1f}%", num_style),
                Paragraph(f"{r.get('Delay Coverage Rate (%)', 0):.1f}%", num_style),
                Paragraph(f"{int(r.get('Estimated Averted Delayed Flights', 0)):,}", num_style),
                Paragraph(f"{r.get('Gross Delay Minutes Saved', 0):,.0f}", num_style),
                Paragraph(f"{net_val:,.0f}", net_style),
            ])

        r_col_w = [0.40 * inch, 0.50 * inch, 0.46 * inch, 0.46 * inch, 0.80 * inch, 0.78 * inch]
        rt = Table(resched_data, colWidths=r_col_w, repeatRows=1)
        resched_style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff5f5")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]
        # Highlight tau=0.171 row (data row index 4 = table row 5) only if enough rows
        if len(resched_data) > 5:
            resched_style_cmds.append(("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#fff0cc")))
        rt.setStyle(TableStyle(resched_style_cmds))
        story.append(rt)
        story.append(Spacer(1, 2))

        if "rescheduling_tradeoff" in figures_paths and figures_paths["rescheduling_tradeoff"].exists():
            story.append(Image(str(figures_paths["rescheduling_tradeoff"]), width=3.5 * inch, height=1.4 * inch))
            story.append(Paragraph(
                "<b>Fig. 3.</b> Policy sweep: Delay coverage (%) vs. intervention rate (%) and net minutes saved. "
                "Net benefit is negative at all evaluated thresholds (simulation estimates under linear friction model).",
                body_style))
            story.append(Spacer(1, 2))

        story.append(Paragraph("VIII. CONCLUSION", h1_style))
        story.append(Paragraph(
            "Phase 2 unified leakage-safe feature engineering with gradient boosting and deep learning. "
            "The Deep Tabular MLP attained the highest discrimination (ROC-AUC 0.6196, PR-AUC 0.2319), "
            "while Platt calibration reduced ECE from 0.2255 to 0.0389. "
            "Rescheduling simulations demonstrate that proactive interventions impose systemic schedule friction "
            "yielding <font color='#cc0000'>negative net benefit</font> under linear penalty models, "
            "underscoring the need for accurate delay probability estimation before operational deployment.",
            body_style))

        story.append(Paragraph("REFERENCES", h1_style))
        story.append(Paragraph(
            "[1] U.S. BTS, 'Airline On-Time Performance Data (2015),' 2015.<br/>"
            "[2] C. Guo et al., 'On Calibration of Modern Neural Networks,' ICML, 2017.<br/>"
            "[3] T. Chen &amp; C. Guestrin, 'XGBoost: A Scalable Tree Boosting System,' KDD, 2016.<br/>"
            "[4] G. Ke et al., 'LightGBM: A Highly Efficient GBDT,' NeurIPS, 2017.<br/>"
            "[5] C. Guo &amp; F. Berkhahn, 'Entity Embeddings of Categorical Variables,' arXiv, 2016.",
            body_style))

        doc.build(story, canvasmaker=NumberedCanvasIEEE)
        logger.info(f"Built IEEE two-column PDF document at {output_pdf_path}.")
        return output_pdf_path
