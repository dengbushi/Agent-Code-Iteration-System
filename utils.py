from pathlib import Path
import io
import pandas as pd


def save_run(cfg, journal):
    best_node = journal.get_best_node(only_good=False)
    if best_node is not None:
        with open("best_solution.py", "w", encoding="utf-8") as f:
            f.write(best_node.code)

    good_nodes = journal.get_good_nodes()
    for i, node in enumerate(good_nodes):
        filename = f"good_solution_{i}.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(node.code)


def preview_csv(p: Path) -> str:
    df = pd.read_csv(p)
    out: list[str] = []
    out.append(f"--- Data Preview for {str(p)} ---")
    out.append(f"Shape: {df.shape[0]} rows and {df.shape[1]} columns.")
    out.append("-" * 20)
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=False)
    info_str = buffer.getvalue()
    out.append("Column Overview (Name, Non-Null Count, Dtype):")
    out.append(info_str)
    out.append("-" * 20)
    out.append("Data Head:")
    out.append(df.head().to_string())
    out.append("-" * 20)
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        out.append("Numerical Columns Statistics:")
        out.append(numeric_df.describe().to_string())
        out.append("-" * 20)
    if 'sample_submission.csv' in str(p):
        out.append("Submission File Format:")
        out.append("The goal is to predict the 'positive' column for each 'id'.")
    out.append(f"--- End of Preview for {str(p)} ---")
    return "\n\n".join(out)


def data_preview_generate(base_path):
    result = []
    files = [p for p in Path(base_path).iterdir()]
    for f in sorted(files):
        result.append(preview_csv(f))
    return "\n\n".join(result)


