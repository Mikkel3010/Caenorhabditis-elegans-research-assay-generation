import json
import os
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ASSAYS_DIR = Path("assay_extractions")

SYSTEM_PROMPT = """You are a deduplication tool for a C. elegans assay database. You are given a CSV file containing assay descriptions. Your job is to identify groups of assays that are essentially the same measurement, even if they differ in minor details such as wording, source paper, genetic background, or developmental stage.

Two assays are "essentially the same" if they measure the same observable quantity, in the same tissue or cell type, at the same resolution level, using the same equipment. Minor differences that do NOT make assays distinct include:
- Different genetic backgrounds or mutant strains used
- Different developmental stages scored (e.g. L1 vs L4) unless the biology is fundamentally different
- Different source papers describing the same protocol
- Slightly different wording of the assay name
- Different throughput or sample sizes

Differences that DO make assays distinct include:
- Different quantities measured (e.g. speed vs reversal rate, even if from the same tracking system)
- Different tissues or cell types (e.g. pharyngeal pumping vs body wall muscle contraction)
- Different resolution levels (e.g. whole-organism behavior vs subcellular imaging)
- Different equipment requirements that reflect a fundamentally different method (e.g. DIC microscopy vs electron microscopy for corpse scoring)

Output format:

Return a JSON array of objects, one per duplicate group. Each object has:

- group_id: integer, starting at 1
- keep: the assay_name of the single best representative to retain (prefer the entry with the most complete metadata, highest throughput, or most widely cited source paper)
- drop: array of assay_name strings that are duplicates of the kept entry
- justification: one sentence explaining why these are the same assay

Only include groups where there is at least one duplicate to drop. Assays that are unique should not appear in the output.

Output only the JSON array — no prose, no markdown fences."""

USER_PROMPT = """Identify duplicate assays in the following CSV file. For each group of duplicates, indicate which one to keep and which to drop.

Assay database (CSV):
{csv_content}"""


def dedup(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    csv_content = df.to_csv(index=False)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_PROMPT.format(csv_content=csv_content),
        }],
    )

    groups = json.loads(response.content[0].text)

    to_drop = {name for g in groups for name in g["drop"]}
    deduped = df[~df["assay_name"].isin(to_drop)].reset_index(drop=True)

    print(f"Found {len(groups)} duplicate groups.")
    print(f"Dropping {len(to_drop)} assays, keeping {len(deduped)} of {len(df)}.")
    for g in groups:
        print(f"  [{g['group_id']}] keep='{g['keep']}' | drop={g['drop']}")
        print(f"       {g['justification']}")

    return deduped


if __name__ == "__main__":
    csv_path = ASSAYS_DIR / "cytoplasm.csv"
    deduped_df = dedup(csv_path)

    out_path = csv_path.with_stem(csv_path.stem + "_deduped")
    deduped_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
