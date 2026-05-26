import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from collections import deque

MAX_REQUESTS = 10
WINDOW_SECONDS = 60
SAFETY_SECONDS = 1.0

request_times = deque()

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"


QUERY = """
query VariantsInGene($geneSymbol: String!) {
  gene(gene_symbol: $geneSymbol, reference_genome: GRCh38) {
    gene_id
    symbol

    clinvar_variants {
      variant_id
      clinical_significance
    }

    variants(dataset: gnomad_r4) {
      variant_id
      pos
      hgvs
      hgvsc
      hgvsp
      consequence
      exome {
        ac
        ac_hemi
        ac_hom
        an
        af
      }
    }
  }
}
"""


def wait_for_rate_limit():
    now = time.time()

    while request_times and now - request_times[0] >= WINDOW_SECONDS:
        request_times.popleft()

    if len(request_times) >= MAX_REQUESTS:
        sleep_time = WINDOW_SECONDS - (now - request_times[0]) + SAFETY_SECONDS
        # print(f"Rate limit reached. Sleeping {sleep_time:.1f} seconds...")
        time.sleep(sleep_time)

    request_times.append(time.time())


def post_gnomad(payload, gene_symbol, max_retries=5):
    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(max_retries):
        wait_for_rate_limit()

        try:
            r = requests.post(GNOMAD_API_URL, json=payload, timeout=(10, 300))
        except requests.exceptions.RequestException as e:
            wait_time = min(2**attempt * 5, 60)
            print(
                f"Request error for {gene_symbol}: {e}. Sleeping {wait_time} seconds..."
            )
            time.sleep(wait_time)
            continue

        if r.status_code == 200:
            return r

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")

            if retry_after is not None:
                wait_time = int(retry_after)
            else:
                wait_time = 60

            print(f"HTTP 429 for {gene_symbol}. Sleeping {wait_time} seconds...")
            time.sleep(wait_time)
            continue

        if r.status_code in retryable_statuses:
            wait_time = min(2**attempt * 5, 60)
            print(
                f"HTTP {r.status_code} for {gene_symbol}. Sleeping {wait_time} seconds before retry..."
            )
            time.sleep(wait_time)
            continue

        return r

    print(f"Failed after {max_retries} retries for {gene_symbol}")
    return r


def fetch_gene(gene_symbol):
    payload = {"query": QUERY, "variables": {"geneSymbol": gene_symbol}}

    r = post_gnomad(payload, gene_symbol)

    if r.status_code != 200:
        print(f"HTTP {r.status_code} for {gene_symbol}")
        return pd.DataFrame()

    if not r.text.strip():
        print(f"Empty response for {gene_symbol}")
        return pd.DataFrame()

    try:
        res = r.json()
    except ValueError:
        print(f"Invalid JSON for {gene_symbol}")
        print("Response preview:", r.text[:200])
        return pd.DataFrame()

    if "errors" in res or res.get("data", {}).get("gene") is None:
        print(f"error for {gene_symbol}: {res.get('errors')}")
        return pd.DataFrame()

    gene = res["data"]["gene"]

    clinvar = pd.json_normalize(gene.get("clinvar_variants") or [])
    variants = pd.json_normalize(gene.get("variants") or [])

    if clinvar.empty:
        return pd.DataFrame()

    clinvar["gene_symbol"] = gene_symbol
    clinvar["gene_id"] = gene.get("gene_id")

    if not variants.empty:
        variants["source"] = "exome"
        variants["allele_count"] = variants.get("exome.ac")
        variants["allele_number"] = variants.get("exome.an")
        variants["allele_frequency"] = variants.get("exome.af")
        variants["n_homozygotes"] = variants.get("exome.ac_hom")
        variants["n_hemizygotes"] = variants.get("exome.ac_hemi")

        keep = [
            "variant_id",
            "pos",
            "hgvs",
            "hgvsc",
            "hgvsp",
            "consequence",
            "source",
            "allele_count",
            "allele_number",
            "allele_frequency",
            "n_homozygotes",
            "n_hemizygotes",
        ]
        keep = [c for c in keep if c in variants.columns]

        variants = variants[keep].drop_duplicates("variant_id")
        clinvar = clinvar.merge(variants, on="variant_id", how="left")

    return clinvar


def add_variants(variant_df, variant_database):
    genes = (
        variant_df["Human_Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
    )

    if os.path.exists(variant_database):
        os.remove(variant_database)

    first_write = True

    for gene_symbol in tqdm(genes, total=len(genes), desc="Processing genes"):
        df_gene = fetch_gene(gene_symbol)

        if not df_gene.empty:
            df_out = variant_df.merge(
                df_gene, left_on="Human_Symbol", right_on="gene_symbol", how="left"
            )

            df_out.to_csv(
                variant_database,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            first_write = False
