# ----------------------------------------------------------------------------
# Copyright (c) 2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import os
from pathlib import Path
import shutil
from typing import Union

import pandas as pd

from q2_annotate.busco.types import BUSCOResultsDirectoryFormat
from q2_types.genome_data import GenesDirectoryFormat, ProteinsDirectoryFormat
from q2_annotate.busco.extract_orthologs import SequenceType


def collate_busco_results(
    results: BUSCOResultsDirectoryFormat,
) -> BUSCOResultsDirectoryFormat:
    collated_results = BUSCOResultsDirectoryFormat()

    result_dfs = []
    for result in results:
        df = pd.read_csv(result.path / "busco_results.tsv", sep="\t", index_col=0)
        result_dfs.append(df)

    pd.concat(result_dfs).to_csv(
        os.path.join(collated_results.path, "busco_results.tsv"),
        sep="\t",
        index=True,
        header=True,
    )

    return collated_results


def _merge_usco_dirs(
    usco_dirs: list[Union[GenesDirectoryFormat, ProteinsDirectoryFormat]],
    seq_type: SequenceType,
) -> Union[GenesDirectoryFormat, ProteinsDirectoryFormat]:
    """Merge multiple USCO directories into a single directory.

    Args:
        usco_dirs: A list of USCO directory formats to merge.
        seq_type (str): Name of the sequence type to extract. Must be `nucleotide` or
            `protein`.

    Returns:
        A new USCO directory format containing all sequences from the input directories.
    """
    if seq_type.is_protein():
        merged_dir = ProteinsDirectoryFormat()
    else:
        merged_dir = GenesDirectoryFormat()

    for usco_dir in usco_dirs:
        for fp in Path(str(usco_dir)).glob("*.fasta"):
            dest = Path(str(merged_dir)) / fp.name
            if dest.exists():
                with open(fp) as src, open(dest, "a") as dst:
                    for line in src:
                        dst.write(line)
            else:
                shutil.copy(fp, dest)

    return merged_dir


def collate_busco_sequences(
    dna_sequences: GenesDirectoryFormat,
    protein_sequences: ProteinsDirectoryFormat,
) -> (GenesDirectoryFormat, ProteinsDirectoryFormat):  # type:ignore

    merged_nucl = _merge_usco_dirs(list(dna_sequences), SequenceType.NUCLEOTIDE)
    merged_prot = _merge_usco_dirs(list(protein_sequences), SequenceType.PROTEIN)

    return merged_nucl, merged_prot
