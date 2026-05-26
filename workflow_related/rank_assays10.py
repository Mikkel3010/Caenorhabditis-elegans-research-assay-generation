import os
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ASSAYS_DIR = Path("assay_extractions")

SYSTEM_PROMPT = """You are an assay recommendation tool for C. elegans experiments. You are given information about a gene with a suspected pathogenic missense variant, and a database of available assays. Your job is to rank the top 10 assays most likely to detect a phenotypic consequence of the variant.

Your reasoning should consider:

1. What biological systems and tissues the gene is involved in, based on the gene description and known orthologs.
2. What functional consequences the specific amino acid substitution might have, considering the properties of the original and substituted amino acids and the position in the protein.
3. Which assays in the database are most likely to detect a phenotypic change caused by this variant, based on the match between the gene's biology and each assay's sensitivity.
4. Practical considerations: if two assays are equally relevant biologically, rank the one with lower equipment requirements and higher throughput above the other.

Output format:

Return a JSON array of exactly 10 objects, ranked from best (index 0) to worst (index 9). Each object must have these fields:

- rank: integer, 1-10
- assay_name: exact assay name as it appears in the database
- confidence: one of "HIGH", "MEDIUM", or "LOW". HIGH means the gene's biology strongly and directly matches the assay. MEDIUM means there is a reasonable but indirect connection. LOW means the recommendation is speculative due to limited information.
- reasoning: 2-4 sentences explaining why this assay is ranked here. Reference specific properties of the gene, the variant, and the assay.

Output only the JSON array — no prose, no markdown fences."""

USER_PROMPT = """Given the following gene information and assay database, rank the top 10 assays most likely to detect whether the missense variant is pathogenic.

Gene name: {gene_name}

Gene description: {gene_description}

Protein sequence (FASTA):
{fasta_sequence}

Missense variant: {variant_description}

Available assays (CSV):
{csv_content}"""

def recommend_assay(
    gene_name: str,
    gene_description: str,
    fasta_sequence: str,
    variant_description: str,
    csv_path: Path,
) -> str:
    df = pd.read_csv(csv_path)
    csv_content = df.to_csv(index=False)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=10000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_PROMPT.format(
                gene_name=gene_name,
                gene_description=gene_description,
                fasta_sequence=fasta_sequence,
                variant_description=variant_description,
                csv_content=csv_content,
            ),
        }],
    )
    return response.content[0].text


if __name__ == "__main__":
    result = recommend_assay(
        gene_name="ubq-1",
        gene_description="Predicted to enable protein tag activity. Predicted to be a structural constituent of ribosome. Involved in programmed cell death involved in cell development and ubiquitin-dependent protein catabolic process. Predicted to be located in cytoplasm. Predicted to be active in cytosolic ribosome and nucleus. Is expressed in body wall musculature; hypodermis; and neurons. Orthologous to human UBC (ubiquitin C).",
        fasta_sequence="MQIFVKTLTGKTITLEVEASDTIENVKAKIQDKEGIPPDQQR...",
        variant_description="Leucine (L) -> Phenylalanine (F) at position 167",
        csv_path=ASSAYS_DIR / "cytoplasm.csv",
    )
    print(result)
