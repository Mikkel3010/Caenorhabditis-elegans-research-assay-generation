import pandas as pd


def alphamissense_matching(
    variant_df,
    alph_path=r"downloaded_data\AlphaMissense_aa_substitutions.tsv",
    chunksize=1_000_000,
):
    wanted_uniprots = set(variant_df["uniprot_id"].dropna().astype(str).str.strip())

    chunks = []

    for chunk in pd.read_csv(
        alph_path,
        sep="\t",
        comment="#",
        chunksize=chunksize,
        dtype=str,
    ):
        chunk["uniprot_id"] = chunk["uniprot_id"].str.strip()
        chunk = chunk[chunk["uniprot_id"].isin(wanted_uniprots)]

        if not chunk.empty:
            chunks.append(chunk)

    alphamissense_subset_df = pd.concat(chunks, ignore_index=True)

    variant_df = variant_df.copy()
    variant_df["uniprot_id"] = variant_df["uniprot_id"].astype(str).str.strip()
    variant_df["short_hgvsp"] = variant_df["short_hgvsp"].astype(str).str.strip()

    alphamissense_subset_df["uniprot_id"] = (
        alphamissense_subset_df["uniprot_id"].astype(str).str.strip()
    )

    alphamissense_subset_df["protein_variant"] = (
        alphamissense_subset_df["protein_variant"].astype(str).str.strip()
    )

    matched_variant_df = variant_df.merge(
        alphamissense_subset_df,
        left_on=["uniprot_id", "short_hgvsp"],
        right_on=["uniprot_id", "protein_variant"],
        how="left",
    )

    return matched_variant_df, alphamissense_subset_df
