import re
import gzip
import pandas as pd
from pathlib import Path
from Bio.SeqIO.FastaIO import SimpleFastaParser


def fasta_data_to_fasta_df(fasta_path, species):
    fasta_path = Path(fasta_path)
    records = []

    species = species.lower()

    opener = gzip.open if fasta_path.suffix == ".gz" else open

    with opener(fasta_path, "rt", encoding="utf-8") as handle:
        for header, sequence in SimpleFastaParser(handle):
            row = {
                "FASTA ID": header.split()[0],
                "Header": header,
                "FASTA Sequence": sequence,
            }

            if species in ["c_elegans"]:
                # Wormbase headers are: key=value
                matches = re.findall(r'(\w+)=("[^"]+"|\S+)', header)

                for key, value in matches:
                    row[key] = value.strip('"')

            elif species in ["human"]:
                # Ensembl headers are : key:value
                matches = re.findall(r"\b(\w+):([^\s]+)", header)

                for key, value in matches:
                    row[key] = value

                # Fix fields whose value contains spaces
                for key in list(row.keys()):
                    pattern = rf"\b{re.escape(key)}:(.*?)(?=\s+\w+:|$)"
                    match = re.search(pattern, header)

                    if match:
                        value = match.group(1).strip()
                        if " " in value:
                            row[key] = value

            else:
                raise ValueError("human or celegans")

            records.append(row)

    df = pd.DataFrame(records)

    if species == "human":
        df.rename(columns={"Acc": "HGNC"}, inplace=True)
        df["HGNC"] = df["HGNC"].str.replace("]", "", regex=False)

    return df


def keep_reviewed_longest_worm_per_gene(
    df,
    seq_col="FASTA Sequence",
    gene_col="gene",
    status_col="status",
    reviewed_value="Confirmed",
):
    out = df.copy()

    out = out[out[status_col].str.lower().eq(reviewed_value.lower())].copy()

    # regex is for linebreaks and spaces
    out["Worm_protein_length"] = (
        out[seq_col].astype(str).str.replace(r"\s+", "", regex=True).str.len()
    )

    out = (
        out.sort_values([gene_col, "Worm_protein_length"], ascending=[True, False])
        .drop_duplicates(subset=gene_col, keep="first")
        .reset_index(drop=True)
    )

    return out


def keep_longest_seq_per_human_uniprot(
    df,
    seq_col="FASTA Sequence",
    uniprot_col="uniprot_id",
):
    out = df.copy()

    out["Human_protein_length"] = (
        out[seq_col].astype(str).str.replace(r"\s+", "", regex=True).str.len()
    )

    out = (
        out.sort_values([uniprot_col, "Human_protein_length"], ascending=[True, False])
        .drop_duplicates(subset=uniprot_col, keep="first")
        .reset_index(drop=True)
    )

    return out
