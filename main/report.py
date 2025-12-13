import json
import pandas as pd
from pathlib import Path

def save_report(results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "results.csv", index=False)

    summary = df.mean(numeric_only=True).to_dict()
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return df, summary
