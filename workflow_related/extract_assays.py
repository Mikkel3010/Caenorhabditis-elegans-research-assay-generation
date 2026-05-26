import base64
import json
import os
import time
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# PAPERS_DIR = Path("papers/anatomical_structure_development")
# PAPERS_DIR = Path("papers/cellular_process")
PAPERS_DIR = Path("papers/cytoplasm2")
# PAPERS_DIR = Path("papers/intracellular_membraneless_organelle")
# PAPERS_DIR = Path("papers/nucleus")
# PAPERS_DIR = Path("papers/protein_metabolic_process")

OUTPUT_FILE = Path("cytoplasm2.json")

print(PAPERS_DIR)

SYSTEM_PROMPT = """You are an assay extraction tool. You are given scientific papers about C. elegans experiments. Your job is to extract every distinct assay described in the paper.

Only extract assays that are performed on C. elegans. If the paper includes assays on other organisms (e.g. Drosophila, mouse, human cells), skip those.

Each distinct measurement counts as a separate assay even if performed in the same experiment session. For example, if a tracking system simultaneously measures speed, reversal rate, and sinusoidal wavelength, those are three separate assays.

Do not extract general molecular biology methods (e.g. CRISPR strain generation, RNA extraction, Western blot for protein verification) unless they are used as a phenotypic readout to compare mutant vs. wild-type.

For each assay, fill in the following fields. If a field cannot be determined from the paper, use null.

Fields:
- assay_name: A short descriptive identifier (e.g. "Pharyngeal pumping rate", "Paraquat survival", "URX calcium imaging").
- biological_system: The pathway or process the assay probes (e.g. neuromuscular function, mitochondrial health, calcium signaling, reproductive fitness, stress response, gene regulation, axon guidance).
- tissue_cell_type: Where the readout comes from (e.g. whole organism, pharynx, specific neuron name, body wall muscle, intestine, vulval muscles).
- resolution_level: One of: whole-organism behavior, cellular event, molecular readout, subcellular structure.
- what_is_measured: The specific observable quantity (e.g. speed in μm/s, offspring count, GFP fluorescence intensity, YFP/CFP ratio, survival percentage).
- output_type: One of: continuous value, proportion, count, binary, distribution, time series.
- equipment_required: What the lab needs to run this assay (e.g. dissecting microscope, fluorescence microscope, tracking camera + software, FACS sorter, electron microscope, qPCR machine).
- requires_transgenic_strain: true or false. If true, briefly state what kind in parentheses (e.g. "true (GFP reporter for gene X)").
- throughput: How many animals per run and approximate time per run, as described in the paper.
- source_paper: Full citation in Vancouver format (e.g. "Smith J, Jones A, Lee B. Title of paper. Journal Name. 2020;1(2):100-110.").

Output a JSON array where each element is one assay object with exactly these 10 keys. Output only the JSON — no prose, no markdown fences."""

USER_PROMPT = "Extract all C. elegans assays from the attached paper and return them as a JSON array as specified in your instructions."


def extract_assays(client: anthropic.Anthropic, pdf_path: Path, retries: int = 5) -> str:
    pdf_data = base64.standard_b64encode(pdf_path.read_bytes()).decode("utf-8")
    for attempt in range(1, retries + 1):
        try:
            with client.messages.stream(
                model="claude-opus-4-7",
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_data,
                            },
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                }],
            ) as stream:
                return stream.get_final_message().content[0].text
        except anthropic.RateLimitError:
            if attempt < retries:
                wait = 65
                print(f"\n  Rate limited — waiting {wait}s (attempt {attempt}/{retries})...", end=" ", flush=True)
                time.sleep(wait)
            else:
                raise


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {PAPERS_DIR}")
    print("=" * 60)

    all_assays = []

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf_path.name}")
        print(f"  Size: {pdf_path.stat().st_size / 1024:.1f} KB")

        try:
            print(f"  Sending to Claude...", end=" ", flush=True)
            raw_output = extract_assays(client, pdf_path)
            print("done")

            assays = json.loads(raw_output)
            for assay in assays:
                assay["source_file"] = pdf_path.name
            all_assays.extend(assays)
            print(f"  Extracted: {len(assays)} assays")

        except json.JSONDecodeError:
            print(f"\n  FAILED to parse JSON — raw output saved")
            all_assays.append({"source_file": pdf_path.name, "parse_error": raw_output})
        except Exception as e:
            print(f"\n  FAILED: {e}")
            all_assays.append({"source_file": pdf_path.name, "error": str(e)})

        time.sleep(65)

    print("\n" + "=" * 60)
    # OUTPUT_FILE.write_text(json.dumps(all_assays, indent=2))
    # print(f"Saved {len(all_assays)} assays to {OUTPUT_FILE}")

    df = pd.DataFrame(all_assays)
    csv_path = Path("assay_extractions") / OUTPUT_FILE.with_suffix(".csv").name
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved flat CSV to {csv_path}")


if __name__ == "__main__":
    main()
