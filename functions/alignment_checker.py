import re
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1


def parse_protein_hgvs(protein_hgvs_value):
    # p.Asp29Tyr to fx: (D,29,Y): one letter codings and position
    match = re.fullmatch(
        r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|Ter)",
        str(protein_hgvs_value).strip(),
    )

    (
        reference_amino_acid_three_letter,
        amino_acid_position,
        alternate_amino_acid_three_letter,
    ) = match.groups()

    reference_amino_acid = seq1(
        reference_amino_acid_three_letter,
        custom_map={"Ter": "*"},
        undef_code="X",
    )

    alternate_amino_acid = seq1(
        alternate_amino_acid_three_letter,
        custom_map={"Ter": "*"},
        undef_code="X",
    )

    return reference_amino_acid, int(amino_acid_position), alternate_amino_acid


def check_conserved_human_variant_to_worm(
    df,
    human_seq_col="Human_FASTA Sequence",
    worm_seq_col="Worm_FASTA Sequence",
    hgvsp_col="hgvsp",
):
    df = df.copy()

    aligner = PairwiseAligner(scoring="blastp")
    aligner.mode = "global"

    cache = {}

    parsed = df[hgvsp_col].apply(parse_protein_hgvs)
    df["human_ref_aa"] = parsed.apply(lambda parsed_value: parsed_value[0])
    df["aa_position"] = parsed.apply(lambda parsed_value: parsed_value[1])
    df["human_alt_aa"] = parsed.apply(lambda parsed_value: parsed_value[2])

    df["short_hgvsp"] = (
        df["human_ref_aa"].astype(str)
        + df["aa_position"].astype("Int64").astype(str)
        + df["human_alt_aa"].astype(str)
    )

    def map_row(row):
        human_seq = str(row[human_seq_col])
        worm_seq = str(row[worm_seq_col])
        human_idx = int(row["aa_position"]) - 1

        # small FASTA/reference check
        if human_idx < 0 or human_idx >= len(human_seq):
            return pd.Series(
                {
                    "alignment": pd.NA,
                    "alignment_indices": pd.NA,
                    "alignment_col": pd.NA,
                    "worm_position": pd.NA,
                    "human_aa": pd.NA,
                    "worm_aa": pd.NA,
                    "is_aligned": False,
                    "is_conserved": pd.NA,
                    "hgvs_ref_matches_human": False,
                    "ref_check_error": "AA position outside human FASTA sequence",
                }
            )

        human_aa = human_seq[human_idx]

        if row["human_ref_aa"] != human_aa:
            return pd.Series(
                {
                    "alignment": pd.NA,
                    "alignment_indices": pd.NA,
                    "alignment_col": pd.NA,
                    "worm_position": pd.NA,
                    "human_aa": human_aa,
                    "worm_aa": pd.NA,
                    "is_aligned": False,
                    "is_conserved": pd.NA,
                    "hgvs_ref_matches_human": False,
                    "ref_check_error": f"HGVS ref {row['human_ref_aa']} != FASTA aa {human_aa}",
                }
            )

        key = (human_seq, worm_seq)

        if key not in cache:
            cache[key] = aligner.align(human_seq, worm_seq)[0]

        alignment = cache[key]
        indices = alignment.indices

        matching_cols = (indices[0] == human_idx).nonzero()[0]

        if len(matching_cols) == 0:
            return pd.Series(
                {
                    "alignment": alignment,
                    "alignment_indices": indices,
                    "alignment_col": pd.NA,
                    "worm_position": pd.NA,
                    "human_aa": human_aa,
                    "worm_aa": pd.NA,
                    "is_aligned": False,
                    "is_conserved": pd.NA,
                    "hgvs_ref_matches_human": True,
                    "ref_check_error": pd.NA,
                }
            )

        alignment_col = matching_cols[0]
        worm_idx = indices[1, alignment_col]

        worm_aa = worm_seq[worm_idx] if worm_idx != -1 else pd.NA

        is_aligned = worm_idx != -1
        is_conserved = human_aa == worm_aa if is_aligned else pd.NA

        return pd.Series(
            {
                "alignment": alignment,
                "alignment_indices": indices,
                "alignment_col": alignment_col,
                "worm_position": worm_idx + 1 if is_aligned else pd.NA,
                "human_aa": human_aa,
                "worm_aa": worm_aa,
                "is_aligned": is_aligned,
                "is_conserved": is_conserved,
                "hgvs_ref_matches_human": True,
                "ref_check_error": pd.NA,
            }
        )

    mapped = df.apply(map_row, axis=1)

    return pd.concat([df, mapped], axis=1)
