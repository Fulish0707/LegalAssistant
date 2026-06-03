import html
import json
from itertools import combinations

import pandas as pd


ALLOWED_STATISTICS_EXTENSIONS = {"csv", "tsv", "xlsx"}
MAX_ANALYSIS_ROWS = 100000
MAX_AI_SAMPLE_ROWS = 20


def load_table(file_path, extension):
    extension = extension.lower()
    if extension == "csv":
        return pd.read_csv(file_path)
    if extension == "tsv":
        return pd.read_csv(file_path, sep="\t")
    if extension == "xlsx":
        return pd.read_excel(file_path, engine="openpyxl")
    raise ValueError("Unsupported file type. Upload CSV, TSV, or XLSX.")


def _safe_number(value):
    if pd.isna(value):
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _json_preview(data, limit=12000):
    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


def analyze_dataframe(df):
    if df.empty:
        raise ValueError("The uploaded file has no rows to analyze.")

    if len(df) > MAX_ANALYSIS_ROWS:
        df = df.head(MAX_ANALYSIS_ROWS).copy()

    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(exclude="number")

    columns = []
    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        columns.append({
            "name": str(column),
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_percent": round((missing_count / len(df)) * 100, 2),
            "unique_count": int(series.nunique(dropna=True)),
        })

    numeric_summary = []
    if not numeric_df.empty:
        describe = numeric_df.describe().transpose()
        for column, row in describe.iterrows():
            numeric_summary.append({
                "column": str(column),
                "count": _safe_number(row.get("count")),
                "mean": _safe_number(row.get("mean")),
                "std": _safe_number(row.get("std")),
                "min": _safe_number(row.get("min")),
                "q25": _safe_number(row.get("25%")),
                "median": _safe_number(row.get("50%")),
                "q75": _safe_number(row.get("75%")),
                "max": _safe_number(row.get("max")),
            })

    categorical_summary = []
    for column in categorical_df.columns[:20]:
        value_counts = categorical_df[column].fillna("(missing)").astype(str).value_counts().head(5)
        categorical_summary.append({
            "column": str(column),
            "top_values": [
                {"value": str(value), "count": int(count)}
                for value, count in value_counts.items()
            ],
        })

    correlations = []
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr(numeric_only=True)
        for left, right in combinations(corr.columns, 2):
            value = corr.loc[left, right]
            if pd.notna(value):
                correlations.append({
                    "left": str(left),
                    "right": str(right),
                    "correlation": _safe_number(value),
                })
        correlations.sort(key=lambda item: abs(item["correlation"] or 0), reverse=True)
        correlations = correlations[:10]

    warnings = []
    if len(df) == MAX_ANALYSIS_ROWS:
        warnings.append(f"Analysis was capped at the first {MAX_ANALYSIS_ROWS:,} rows.")
    if df.isna().any().any():
        warnings.append("Missing values were detected. Review columns with high missing percentages before drawing conclusions.")
    if numeric_df.empty:
        warnings.append("No numeric columns were detected, so numeric statistics and correlations are unavailable.")

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlations": correlations,
        "warnings": warnings,
        "sample_rows": df.head(MAX_AI_SAMPLE_ROWS).where(pd.notna(df), None).to_dict(orient="records"),
    }


def build_ai_statistics_prompt(filename, analysis, user_question=""):
    return (
        "You are an analyst assistant inside a personal work companion app. "
        "Interpret this dataset analysis for a working professional. "
        "Be specific, practical, and careful about uncertainty. "
        "Do not invent columns or causal claims. "
        "Return concise HTML with sections for Executive Readout, Notable Patterns, Risks or Data Quality, "
        "and Recommended Next Steps.\n\n"
        f"File: {filename}\n"
        f"User focus: {user_question or 'General business/statistical readout'}\n"
        f"Analysis JSON:\n{_json_preview(analysis)}"
    )


def render_statistics_html(filename, analysis, ai_insight=None, ai_enabled=False):
    filename_html = html.escape(filename)
    status = "AI insight included" if ai_enabled and ai_insight else "Local statistical analysis"

    parts = [
        f"<h2>Statistics Analysis: {filename_html}</h2>",
        f"<p><strong>{html.escape(status)}</strong></p>",
        "<div class='row g-3 mb-3'>",
        f"<div class='col-md-4'><div class='stat-tile'><span>Rows</span><strong>{analysis['row_count']}</strong></div></div>",
        f"<div class='col-md-4'><div class='stat-tile'><span>Columns</span><strong>{analysis['column_count']}</strong></div></div>",
        f"<div class='col-md-4'><div class='stat-tile'><span>Numeric fields</span><strong>{len(analysis['numeric_summary'])}</strong></div></div>",
        "</div>",
    ]

    if ai_insight:
        parts.append("<section class='mb-4'>")
        parts.append(ai_insight)
        parts.append("</section>")
    elif not ai_enabled:
        parts.append(
            "<p class='text-muted'>Add GOOGLE_API_KEY in .env to unlock AI interpretation. "
            "The statistical tables below are computed locally.</p>"
        )

    if analysis["warnings"]:
        parts.append("<h3>Data Notes</h3><ul>")
        for warning in analysis["warnings"]:
            parts.append(f"<li>{html.escape(warning)}</li>")
        parts.append("</ul>")

    parts.append("<h3>Columns</h3><div class='table-responsive'><table class='table table-sm table-striped'>")
    parts.append("<thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Missing %</th><th>Unique</th></tr></thead><tbody>")
    for column in analysis["columns"]:
        parts.append(
            "<tr>"
            f"<td>{html.escape(column['name'])}</td>"
            f"<td>{html.escape(column['dtype'])}</td>"
            f"<td>{column['missing_count']}</td>"
            f"<td>{column['missing_percent']}</td>"
            f"<td>{column['unique_count']}</td>"
            "</tr>"
        )
    parts.append("</tbody></table></div>")

    if analysis["numeric_summary"]:
        parts.append("<h3>Numeric Summary</h3><div class='table-responsive'><table class='table table-sm table-striped'>")
        parts.append("<thead><tr><th>Column</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Q1</th><th>Median</th><th>Q3</th><th>Max</th></tr></thead><tbody>")
        for row in analysis["numeric_summary"]:
            parts.append(
                "<tr>"
                f"<td>{html.escape(row['column'])}</td>"
                f"<td>{row['count']}</td>"
                f"<td>{row['mean']}</td>"
                f"<td>{row['std']}</td>"
                f"<td>{row['min']}</td>"
                f"<td>{row['q25']}</td>"
                f"<td>{row['median']}</td>"
                f"<td>{row['q75']}</td>"
                f"<td>{row['max']}</td>"
                "</tr>"
            )
        parts.append("</tbody></table></div>")

    if analysis["correlations"]:
        parts.append("<h3>Strongest Numeric Relationships</h3><div class='table-responsive'><table class='table table-sm table-striped'>")
        parts.append("<thead><tr><th>Field 1</th><th>Field 2</th><th>Correlation</th></tr></thead><tbody>")
        for row in analysis["correlations"]:
            parts.append(
                "<tr>"
                f"<td>{html.escape(row['left'])}</td>"
                f"<td>{html.escape(row['right'])}</td>"
                f"<td>{row['correlation']}</td>"
                "</tr>"
            )
        parts.append("</tbody></table></div>")

    if analysis["categorical_summary"]:
        parts.append("<h3>Top Categorical Values</h3>")
        for row in analysis["categorical_summary"]:
            parts.append(f"<h4>{html.escape(row['column'])}</h4><ul>")
            for item in row["top_values"]:
                parts.append(f"<li>{html.escape(item['value'])}: {item['count']}</li>")
            parts.append("</ul>")

    return "\n".join(parts)


def build_statistics_knowledge_text(filename, analysis, ai_insight=None):
    payload = {
        "filename": filename,
        "row_count": analysis["row_count"],
        "column_count": analysis["column_count"],
        "columns": analysis["columns"],
        "numeric_summary": analysis["numeric_summary"],
        "correlations": analysis["correlations"],
        "warnings": analysis["warnings"],
    }
    text = f"Statistical analysis for {filename}\n{_json_preview(payload, limit=20000)}"
    if ai_insight:
        text += f"\n\nAI interpretation HTML:\n{ai_insight}"
    return text
