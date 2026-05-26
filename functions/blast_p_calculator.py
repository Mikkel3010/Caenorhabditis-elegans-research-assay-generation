import os
import re
import shutil
import subprocess
import pandas as pd


def _clean_seq(seq):
    if pd.isna(seq):
        return None

    seq = str(seq).strip()

    if seq.lower() in {"", "nan", "none", "missing value", "missing"}:
        return None

    seq = re.sub(r"\s+", "", seq)

    return seq if seq else None


def _safe_id(x):
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", str(x))


def blast_p_calculation(
    df,
    human_id="Human_ID",
    gene1_seq_col="Human_FASTA Sequence",
    gene2_id_col="Worm_ID",
    gene2_seq_col="Worm_FASTA Sequence",
    blastp_path="blastp",
    makeblastdb_path="makeblastdb",
    workdir="blast_single_run",
    evalue=1e-5,
    max_target_seqs=100,  # covers max amount of unique ortologs for a single gene
    num_threads=8,
):
    out = df.copy()

    if shutil.which(blastp_path) is None:
        raise RuntimeError(f"Could not find '{blastp_path}'. Install BLAST+ first.")

    if shutil.which(makeblastdb_path) is None:
        raise RuntimeError(
            f"Could not find '{makeblastdb_path}'. Install BLAST+ first."
        )

    os.makedirs(workdir, exist_ok=True)

    query_fasta = os.path.join(workdir, "human_queries.fasta")
    subject_fasta = os.path.join(workdir, "worm_subjects.fasta")
    subject_db = os.path.join(workdir, "worm_db")
    blast_out = os.path.join(workdir, "human_vs_worm.tsv")

    out["_clean_human_seq"] = out[gene1_seq_col].map(_clean_seq)
    out["_clean_worm_seq"] = out[gene2_seq_col].map(_clean_seq)

    out["_blast_human_id"] = out[human_id].map(_safe_id)
    out["_blast_worm_id"] = out[gene2_id_col].map(_safe_id)

    blast_cols = [
        "BLAST_Gene2ID",
        "BLAST_pident",
        "BLAST_align_len",
        "BLAST_evalue",
        "BLAST_bitscore",
    ]

    human_unique = (
        out[["_blast_human_id", "_clean_human_seq"]]
        .dropna(subset=["_clean_human_seq"])
        .drop_duplicates()
        .copy()
    )

    worm_unique = (
        out[["_blast_worm_id", "_clean_worm_seq"]]
        .dropna(subset=["_clean_worm_seq"])
        .drop_duplicates()
        .copy()
    )

    if human_unique.empty or worm_unique.empty:
        for c in blast_cols:
            out[c] = pd.NA

        return out.drop(
            columns=[
                "_clean_human_seq",
                "_clean_worm_seq",
                "_blast_human_id",
                "_blast_worm_id",
            ]
        )

    with open(query_fasta, "w") as f:
        for _, row in human_unique.iterrows():
            f.write(f">{row['_blast_human_id']}\n{row['_clean_human_seq']}\n")

    with open(subject_fasta, "w") as f:
        for _, row in worm_unique.iterrows():
            f.write(f">{row['_blast_worm_id']}\n{row['_clean_worm_seq']}\n")

    subprocess.run(
        [
            makeblastdb_path,
            "-in",
            subject_fasta,
            "-dbtype",
            "prot",
            "-out",
            subject_db,
        ],
        check=True,
    )

    outfmt = (
        "6 qseqid sseqid pident length mismatch gapopen "
        "qstart qend sstart send evalue bitscore"
    )

    subprocess.run(
        [
            blastp_path,
            "-query",
            query_fasta,
            "-db",
            subject_db,
            "-out",
            blast_out,
            "-outfmt",
            outfmt,
            "-evalue",
            str(evalue),
            "-max_target_seqs",
            str(max_target_seqs),
            "-num_threads",
            str(num_threads),
        ],
        check=True,
    )

    if not os.path.exists(blast_out) or os.path.getsize(blast_out) == 0:
        for c in blast_cols:
            out[c] = pd.NA

        return out.drop(
            columns=[
                "_clean_human_seq",
                "_clean_worm_seq",
                "_blast_human_id",
                "_blast_worm_id",
            ]
        )

    cols = [
        "qseqid",
        "sseqid",
        "pident",
        "align_len",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
    ]

    blast_df = pd.read_csv(blast_out, sep="\t", names=cols)

    best_pair_hits = (
        blast_df.sort_values(
            ["qseqid", "sseqid", "bitscore", "evalue"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["qseqid", "sseqid"], keep="first")
        .copy()
    )

    best_pair_hits["BLAST_Gene2ID"] = best_pair_hits["sseqid"]

    best_pair_hits = best_pair_hits.rename(
        columns={
            "qseqid": "_blast_human_id",
            "sseqid": "_blast_worm_id",
            "pident": "BLAST_pident",
            "align_len": "BLAST_align_len",
            "evalue": "BLAST_evalue",
            "bitscore": "BLAST_bitscore",
        }
    )

    best_pair_hits = best_pair_hits[
        [
            "_blast_human_id",
            "_blast_worm_id",
            "BLAST_Gene2ID",
            "BLAST_pident",
            "BLAST_align_len",
            "BLAST_evalue",
            "BLAST_bitscore",
        ]
    ]

    out = out.merge(
        best_pair_hits,
        on=["_blast_human_id", "_blast_worm_id"],
        how="left",
    )

    missing_human = out["_clean_human_seq"].isna()
    missing_worm = out["_clean_worm_seq"].isna()

    out.loc[missing_human | missing_worm, blast_cols] = pd.NA

    out = out.drop(
        columns=[
            "_clean_human_seq",
            "_clean_worm_seq",
            "_blast_human_id",
            "_blast_worm_id",
        ]
    )

    return out
