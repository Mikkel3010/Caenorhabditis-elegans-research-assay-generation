import os
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ASSAYS_DIR = Path("assay_extractions")

SYSTEM_PROMPT = """You are a lab protocol summarizer for C. elegans experiments. You are given structured information about a single assay and context about the gene and variant being tested. Your job is to write a short, readable experiment description that a lab technician can use to understand what they will be doing and why.

The description should:

1. Start with one sentence stating the goal: what gene/variant is being tested and what phenotype the assay is looking for.
2. Explain in plain language what the assay involves: what the animals do, what gets measured, and how.
3. Note any special requirements: transgenic strains, specific equipment, or reagents.
4. State what a positive result would look like: what difference between mutant and wild-type animals would indicate the variant is pathogenic.
5. Mention approximate scale: how many animals are needed and roughly how long it takes.

Keep the total length to one paragraph of 5-8 sentences. Use plain language. Do not use jargon without briefly explaining it. Do not include raw data or statistical methods."""

USER_PROMPT = """Write a short experiment description for a lab technician based on the following assay information and gene context.

Gene being tested: {gene_name}
Variant: {variant_description}
Reason for testing: The variant is suspected to be pathogenic.

Assay information:
- Assay name: {assay_name}
- Biological system: {biological_system}
- Tissue/cell type: {tissue_cell_type}
- Resolution level: {resolution_level}
- What is measured: {what_is_measured}
- Output type: {output_type}
- Equipment required: {equipment_required}
- Requires transgenic strain: {requires_transgenic}
- Throughput: {throughput}
- Source paper: {source_paper}"""

def run(
    gene_name: str,
    variant_description: str,
    assay_name: str,
    biological_system: str,
    tissue_cell_type: str,
    resolution_level: str,
    what_is_measured: str,
    output_type: str,
    equipment_required: str,
    requires_transgenic: str,
    throughput: str,
    source_paper: str,
) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=10000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_PROMPT.format(
                gene_name=gene_name,
                variant_description=variant_description,
                assay_name=assay_name,
                biological_system=biological_system,
                tissue_cell_type=tissue_cell_type,
                resolution_level=resolution_level,
                what_is_measured=what_is_measured,
                output_type=output_type,
                equipment_required=equipment_required,
                requires_transgenic=requires_transgenic,
                throughput=throughput,
                source_paper=source_paper,
            ),
        }],
    )
    return response.content[0].text


if __name__ == "__main__":
    assays = pd.read_csv(ASSAYS_DIR / "cytoplasm.csv")
    row = assays[assays["assay_name"] == "Linker cell survival scoring"].iloc[0]

    result = run(
        gene_name="ubq-1",
        variant_description="Leucine (L) -> Phenylalanine (F) at position 167",
        assay_name=row["assay_name"],
        biological_system=row["biological_system"],
        tissue_cell_type=row["tissue_cell_type"],
        resolution_level=row["resolution_level"],
        what_is_measured=row["what_is_measured"],
        output_type=row["output_type"],
        equipment_required=row["equipment_required"],
        requires_transgenic=row["requires_transgenic_strain"],
        throughput=row["throughput"],
        source_paper=row["source_paper"],
    )
    print(result)
