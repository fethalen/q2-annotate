# ----------------------------------------------------------------------------
# Copyright (c) 2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import os
from pathlib import Path

import pandas as pd
from q2_types.genome_data import GenesDirectoryFormat, ProteinsDirectoryFormat
from qiime2.plugin.testing import TestPluginBase

from q2_annotate.busco.extract_orthologs import SequenceType
from q2_annotate.busco.partition import (
    _merge_usco_dirs,
    collate_busco_results,
    collate_busco_sequences,
)
from q2_annotate.busco.types import BUSCOResultsDirectoryFormat


class TestBuscoPartition(TestPluginBase):
    package = "q2_annotate.busco.tests"

    def setUp(self):
        super().setUp()

        # USCO sequences
        self.usco_nucl_fp_1 = self.get_data_path("usco_sequences/9802219at2.fna")
        self.usco_nucl_fp_2 = self.get_data_path("usco_sequences/9809052at2.fna")
        self.usco_prot_fp_1 = self.get_data_path("usco_sequences/9802219at2.faa")
        self.usco_prot_fp_2 = self.get_data_path("usco_sequences/9809052at2.faa")

        self.usco_nucl_dir = self.get_data_path("usco_sequences")
        self.usco_prot_dir = self.get_data_path("usco_sequences")
        self.dna_orthos = [
            GenesDirectoryFormat(self.usco_nucl_dir, mode="r"),
            GenesDirectoryFormat(self.usco_nucl_dir, mode="r"),
        ]
        self.prot_orthos = [
            ProteinsDirectoryFormat(self.usco_prot_dir, mode="r"),
            ProteinsDirectoryFormat(self.usco_prot_dir, mode="r"),
        ]

        # BUSCO results data
        self.busco_partition_1 = self.get_data_path("busco_results/partition1")
        self.busco_partition_2 = self.get_data_path("busco_results/partition2")
        self.busco_results = [
            BUSCOResultsDirectoryFormat(self.busco_partition_1, mode="r"),
            BUSCOResultsDirectoryFormat(self.busco_partition_2, mode="r"),
        ]

    def test_collate_busco_results(self):
        collated_busco_result = collate_busco_results(self.busco_results)

        obs = pd.read_csv(os.path.join(str(collated_busco_result), "busco_results.tsv"))
        exp = pd.read_csv(
            self.get_data_path("busco_results/results_all/busco_results.tsv")
        )

        pd.testing.assert_frame_equal(obs, exp)

    def test_collate_busco_sequences(self):
        collated_nucl, collated_prot = collate_busco_sequences(
            self.dna_orthos, self.prot_orthos
        )

        # Check types
        self.assertIsInstance(collated_nucl, GenesDirectoryFormat)
        self.assertIsInstance(collated_prot, ProteinsDirectoryFormat)

    def test_merge_usco_dirs(self):
        merged_nucl = _merge_usco_dirs(self.dna_orthos, SequenceType.NUCLEOTIDE)
        merged_prot = _merge_usco_dirs(self.prot_orthos, SequenceType.PROTEIN)

        self.assertIsInstance(merged_nucl, GenesDirectoryFormat)
        self.assertIsInstance(merged_prot, ProteinsDirectoryFormat)

        nucl_files = list(Path(str(merged_nucl)).glob("*.fna"))
        prot_files = list(Path(str(merged_prot)).glob("*.faa"))

        self.assertEqual(
            {p.name for p in nucl_files},
            {"9802219at2.fna", "9809052at2.fna", "tiny_gene_1.fna"},
        )
        self.assertEqual(
            {p.name for p in prot_files},
            {"9802219at2.faa", "9809052at2.faa"},
        )

        tiny_gene_fp = sorted(nucl_files, key=lambda p: p.name)[2]

        contents = tiny_gene_fp.read_text().strip().splitlines()

        assert contents[0].startswith(">sample1|gene1")
        assert contents[2].startswith(">sample2|gene1")  # appended line
        assert sum(line.startswith(">") for line in contents) == 4
