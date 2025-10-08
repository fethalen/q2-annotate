# ----------------------------------------------------------------------------
# Copyright (c) 2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import glob
import json
import os
import tempfile
from copy import deepcopy
from shutil import copytree
from typing import List, Union

import pandas as pd
import q2templates
from q2_types.feature_data_mag import MAGSequencesDirFmt
from q2_types.genome_data import GenesDirectoryFormat, ProteinsDirectoryFormat
from q2_types.per_sample_sequences import MultiMAGSequencesDirFmt

from q2_annotate._utils import _process_common_input_params, run_command
from q2_annotate.busco.extract_orthologs import (
    DuplicateMode,
    FragmentMode,
    SequenceType,
    _append_uscos,
    _extract_uscos,
    _get_busco_results_dir,
)
from q2_annotate.busco.plots_detailed import _draw_detailed_plots
from q2_annotate.busco.plots_summary import (
    _draw_completeness_vs_contamination,
    _draw_marker_summary_histograms,
    _draw_selectable_summary_histograms,
    _draw_selectable_unbinned_histograms,
)
from q2_annotate.busco.types import BuscoDatabaseDirFmt
from q2_annotate.busco.utils import (
    _calculate_summary_stats,
    _cleanup_bootstrap,
    _extract_json_data,
    _get_feature_table,
    _parse_busco_params,
    _parse_df_columns,
    _partition_dataframe,
    _process_busco_results,
    _validate_lineage_dataset_input,
    _validate_parameters,
    _filter_unbinned_for_partition,
    _add_unbinned_metrics,
)

from q2_annotate._utils import _process_common_input_params, run_command
from q2_annotate.busco.types import BuscoDatabaseDirFmt
from q2_types.sample_data import SampleData
from q2_types.feature_data_mag import MAGSequencesDirFmt
from q2_types.per_sample_sequences import (
    MultiMAGSequencesDirFmt,
    ContigSequencesDirFmt,
    MAGs,
)
import warnings


def _run_busco(input_dir: str, output_dir: str, sample_id: str, params: List[str]):
    """Runs BUSCO on one (sample) directory

    Args:
        input_dir (str): Location where the MAG FASTA files are stored.
        output_dir (str): Location where the final results should be stored.
        sample_id (str): The sample ID.
        params (List[str]): List of parsed arguments to pass to BUSCO.
    """
    base_cmd = ["busco", *params]

    cmd = deepcopy(base_cmd)
    cmd.extend(["--in", input_dir, "--out_path", output_dir, "-o", sample_id])
    run_command(cmd, cwd=os.path.dirname(output_dir))


def _busco_helper(mags, kwargs, additional_metrics):
    results_all = []
    # Get samples directories from MAGs
    if isinstance(mags, MultiMAGSequencesDirFmt):
        sample_dir = mags.sample_dict()
    elif isinstance(mags, MAGSequencesDirFmt):
        sample_dir = {"feature_data": mags.feature_dict()}

    # Filter out all kwargs that are None, False or 0.0
    common_args = _process_common_input_params(
        processing_func=_parse_busco_params, params=kwargs
    )

    with tempfile.TemporaryDirectory() as tmp:
        usco_nucl_dir, usco_prot_dir = GenesDirectoryFormat(), ProteinsDirectoryFormat()

        for sample_id, feature_dict in sample_dir.items():

            _run_busco(
                input_dir=os.path.join(
                    str(mags), "" if sample_id == "feature_data" else sample_id
                ),
                output_dir=str(tmp),
                sample_id=sample_id,
                params=common_args,
            )

            for mag_id, mag_fp in feature_dict.items():
                # Extract and process results from JSON files
                json_path = glob.glob(
                    os.path.join(
                        str(tmp), sample_id, os.path.basename(mag_fp), "*.json"
                    )
                )[0]

                results = _process_busco_results(
                    _extract_json_data(json_path),
                    sample_id,
                    mag_id,
                    os.path.basename(mag_fp),
                    additional_metrics,
                )
                results_all.append(results)

                # Extract nucleotide and protein orthologs
                busco_results_dir = _get_busco_results_dir(tmp, sample_id, mag_fp)
                usco_fps_nucl, usco_fps_prot = (
                    _extract_uscos(
                        busco_results_dir=busco_results_dir,
                        lineage_dataset=kwargs["lineage_dataset"],
                        seq_type=seq_type,
                        min_len=0,
                        min_score=0,
                        fragment_mode=FragmentMode.SKIP,
                        duplicate_mode=DuplicateMode.SKIP,
                    )
                    for seq_type in (SequenceType.NUCLEOTIDE, SequenceType.PROTEIN)
                )

                usco_nucl_dir, usco_prot_dir = (
                    _append_uscos(usco_dir, usco_fps, seq_type, species_tag=mag_id)
                    for usco_dir, usco_fps, seq_type in [
                        (usco_nucl_dir, usco_fps_nucl, SequenceType.NUCLEOTIDE),
                        (usco_prot_dir, usco_fps_prot, SequenceType.PROTEIN),
                    ]
                )

    return pd.DataFrame(results_all), usco_nucl_dir, usco_prot_dir


def _evaluate_busco(
    mags: Union[MultiMAGSequencesDirFmt, MAGSequencesDirFmt],
    db: BuscoDatabaseDirFmt,
    unbinned_contigs: ContigSequencesDirFmt = None,  # NEW unbinned
    mode: str = "genome",
    lineage_dataset: str = None,
    augustus: bool = False,
    augustus_parameters: str = None,
    augustus_species: str = None,
    auto_lineage: bool = False,
    auto_lineage_euk: bool = False,
    auto_lineage_prok: bool = False,
    cpu: int = 1,
    contig_break: int = 10,
    evalue: float = 1e-03,
    limit: int = 3,
    long: bool = False,
    metaeuk_parameters: str = None,
    metaeuk_rerun_parameters: str = None,
    miniprot: bool = False,
    additional_metrics: bool = False,
) -> (pd.DataFrame, GenesDirectoryFormat, ProteinsDirectoryFormat):  # type:ignore
    kwargs = {
        k: v
        for k, v in locals().items()
        if k not in ["mags", "unbinned_contigs", "db", "additional_metrics"]
    }
    kwargs["offline"] = True
    kwargs["download_path"] = str(db)

    if lineage_dataset is not None:
        _validate_lineage_dataset_input(
            lineage_dataset,
            auto_lineage,
            auto_lineage_euk,
            auto_lineage_prok,
            db,
            kwargs,
        )

    # Always call _busco_helper once
    busco_results = _busco_helper(mags, kwargs, additional_metrics)

    # If mags is MultiMAGSequencesDirFmt, add unbinned contigs info
    if isinstance(mags, MultiMAGSequencesDirFmt) and unbinned_contigs:
        busco_results = _add_unbinned_metrics(busco_results, mags, unbinned_contigs)

    return busco_results


def _visualize_busco(output_dir: str, results: pd.DataFrame) -> None:
    results.to_csv(os.path.join(output_dir, "busco_results.csv"), index=False)

    # Outputs different df for sample and feature data
    results = _parse_df_columns(results)
    max_rows = 100

    # Partition data frames
    if len(results["sample_id"].unique()) >= 2:
        counter_col = "sample_id"
        assets_subdir = "sample_data"
        tab_title = ["Sample details", "Feature details"]
        is_sample_data = True

        # Draw selectable histograms (only for sample data mags)
        tabbed_context = {
            "vega_summary_selectable_json": json.dumps(
                _draw_selectable_summary_histograms(results)
            ).replace("NaN", "null"),
        }
    else:
        counter_col = "mag_id"
        tab_title = ["BUSCO plots", "BUSCO table"]
        assets_subdir = "feature_data"
        is_sample_data = False
        tabbed_context = {}

    dfs = _partition_dataframe(results, max_rows, is_sample_data)

    TEMPLATES = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets", "busco"
    )
    templates = [
        os.path.join(TEMPLATES, assets_subdir, file_name)
        for file_name in ["index.html", "detailed_view.html", "table.html"]
    ]
    copytree(
        src=os.path.join(TEMPLATES, assets_subdir), dst=output_dir, dirs_exist_ok=True
    )
    for folder in ["css", "js"]:
        os.makedirs(os.path.join(output_dir, folder))
        copytree(
            src=os.path.join(TEMPLATES, folder),
            dst=os.path.join(output_dir, folder),
            dirs_exist_ok=True,
        )

    # Partition data frames and draw detailed plots
    context = {}
    counter_left = 1
    for i, df in enumerate(dfs):
        count = df[counter_col].nunique()
        counter_right = counter_left + count - 1
        counters = {"from": counter_left, "to": counter_right}
        counter_left += count
        subcontext = _draw_detailed_plots(
            df,
            is_sample_data,
            width=600,
            height=30,
            title_font_size=20,
            label_font_size=17,
            spacing=20,
        )
        context.update(
            {
                f"partition_{i}": {
                    "subcontext": subcontext,
                    "counters": counters,
                    "ids": df[counter_col].unique().tolist(),
                }
            }
        )

    # Render
    vega_json = json.dumps(context).replace("NaN", "null")
    vega_json_summary = json.dumps(_draw_marker_summary_histograms(results)).replace(
        "NaN", "null"
    )
    table_json = _get_feature_table(results)
    stats_json = _calculate_summary_stats(results)

    scatter_json, comp_cont = None, False
    if "completeness" in results.columns and "contamination" in results.columns:
        scatter_json = json.dumps(_draw_completeness_vs_contamination(results)).replace(
            "NaN", "null"
        )
        comp_cont = True

    if (
        "unbinned_contigs" in results.columns
        and "unbinned_contigs_count" in results.columns
    ):
        tabbed_context["vega_selectable_unbinned_json"] = json.dumps(
            _draw_selectable_unbinned_histograms(results)
        ).replace("NaN", "null")
        unbinned = True
    else:
        tabbed_context["vega_selectable_unbinned_json"] = None
        unbinned = False
    tabbed_context.update(
        {
            "tabs": [
                {"title": "QC overview", "url": "index.html"},
                {"title": tab_title[0], "url": "detailed_view.html"},
                {"title": tab_title[1], "url": "table.html"},
            ],
            "vega_json": vega_json,
            "vega_summary_json": vega_json_summary,
            "table": table_json,
            "summary_stats_json": stats_json,
            "scatter_json": scatter_json,
            "comp_cont": comp_cont,
            "unbinned": unbinned,
            "page_size": 100,
        }
    )
    q2templates.render(templates, output_dir, context=tabbed_context)

    # Final cleanup, needed until we fully migrate to Bootstrap 5
    _cleanup_bootstrap(output_dir)


def evaluate_busco(
    ctx,
    mags,
    db,
    unbinned_contigs=None,
    mode="genome",
    lineage_dataset=None,
    augustus=False,
    augustus_parameters=None,
    augustus_species=None,
    auto_lineage=False,
    auto_lineage_euk=False,
    auto_lineage_prok=False,
    cpu=1,
    contig_break=10,
    evalue=1e-03,
    limit=3,
    long=False,
    metaeuk_parameters=None,
    metaeuk_rerun_parameters=None,
    miniprot=False,
    additional_metrics=True,
    num_partitions=None,
):
    _validate_parameters(
        lineage_dataset, auto_lineage, auto_lineage_euk, auto_lineage_prok
    )

    kwargs = {
        k: v
        for k, v in locals().items()
        if k not in ["mags", "unbinned_contigs", "ctx", "db", "num_partitions"]
    }
    _evaluate_busco = ctx.get_action("annotate", "_evaluate_busco")
    collate_busco_results = ctx.get_action("annotate", "collate_busco_results")
    _visualize_busco = ctx.get_action("annotate", "_visualize_busco")
    _filter_contigs = ctx.get_action("assembly", "filter_contigs")
    collate_busco_sequences = ctx.get_action("annotate", "collate_busco_sequences")

    if issubclass(mags.format, MultiMAGSequencesDirFmt):
        partition_action = "partition_sample_data_mags"
    else:
        partition_action = "partition_feature_data_mags"
        if unbinned_contigs is not None:
            warnings.warn(
                "FeatureData[MAG] artifact was provided - "
                "unbinned contigs will be ignored."
            )

    partition_mags = ctx.get_action("types", partition_action)

    (partitioned_mags,) = partition_mags(mags, num_partitions)
    results = []

    if mags.type <= SampleData[MAGs]:
        # we need to match the unbinned contigs to the partitioned MAGs
        for mag_partition in partitioned_mags.values():
            if unbinned_contigs:
                unbinned_filtered = _filter_unbinned_for_partition(
                    unbinned_contigs, mag_partition, _filter_contigs
                )
            else:
                unbinned_filtered = None
            (busco_result,) = _evaluate_busco(
                mag_partition, db, unbinned_filtered, **kwargs
            )
            results.append(busco_result)
    else:
        for mag in partitioned_mags.values():  # each mag is a subset of bins
            (busco_result,) = _evaluate_busco(mag, db, None, **kwargs)
            results.append(busco_result)

    all_nucl_seqs = []
    all_prot_seqs = []

    for mag in partitioned_mags.values():
        busco_result, nucl_seqs, prot_seqs = _evaluate_busco(mag, **kwargs)
        results.append(busco_result)
        all_nucl_seqs.append(nucl_seqs)
        all_prot_seqs.append(prot_seqs)

    (collated_results,) = collate_busco_results(results)
    (visualization,) = _visualize_busco(collated_results)
    collated_nucl_dirs, collated_prot_dirs = collate_busco_sequences(
        all_nucl_seqs, all_prot_seqs
    )

    return collated_results, visualization, collated_nucl_dirs, collated_prot_dirs
