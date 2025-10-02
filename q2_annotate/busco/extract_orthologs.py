import os
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import Union, overload

import pandas as pd
from q2_types.feature_data_mag import MAGSequencesDirFmt
from q2_types.genome_data import GenesDirectoryFormat, ProteinsDirectoryFormat
from q2_types.per_sample_sequences import MultiMAGSequencesDirFmt

from q2_annotate._utils import _process_common_input_params
from q2_annotate.busco.busco import _run_busco
from q2_annotate.busco.types import BuscoDatabaseDirFmt
from q2_annotate.busco.utils import (
    _parse_busco_params,
    _validate_lineage_dataset_input,
    _validate_parameters,
)


class SequenceType(str, Enum):
    NUCLEOTIDE = "nucleotide"
    PROTEIN = "protein"

    def is_nucleotide(self) -> bool:
        return self is SequenceType.NUCLEOTIDE

    def is_protein(self) -> bool:
        return self is SequenceType.PROTEIN


class FragmentMode(str, Enum):
    LONGEST = "longest"
    BEST_SCORE = "best_score"
    SKIP = "skip"


class DuplicateMode(str, Enum):
    LONGEST = "longest"
    BEST_SCORE = "best_score"
    SKIP = "skip"


def _parse_full_table_file(tsv_path: Path) -> pd.DataFrame:
    """
    Parse the full_table.tsv file into a Pandas DataFrame.

    Column names are normalized by replacing spaces with underscores and converting
    uppercase characters to lowercase.

    Expected columns:
        - busco_id
        - status
        - sequence
        - gene_start
        - gene_end
        - strand
        - score
        - length

    Args:
        tsv_path (Path): Path to the BUSCO full_table.tsv file.

    Returns:
        pandas.DataFrame: The complete BUSCO results as a Pandas DataFrame.
    """
    df = pd.read_csv(tsv_path, sep="\t", skiprows=2)
    df.columns = df.columns.str.strip("# ").str.lower().str.replace(" ", "_")
    return df


def _filter_by_mode(df: pd.DataFrame, status: str, mode: Enum) -> pd.DataFrame:
    """Helper to filter a subset of BUSCO rows by mode."""
    subset = df[df["status"] == status]
    if mode == FragmentMode.SKIP or mode == DuplicateMode.SKIP:
        return df.iloc[0:0]
    if mode in {
        FragmentMode.LONGEST,
        FragmentMode.BEST_SCORE,
        DuplicateMode.LONGEST,
        DuplicateMode.BEST_SCORE,
    }:
        key_col = (
            "length"
            if mode in {FragmentMode.LONGEST, DuplicateMode.LONGEST}
            else "score"
        )
        if not subset.empty:
            subset = subset.loc[subset.groupby("busco_id")[key_col].idxmax()]
    return subset


def _filter_full_table_df(
    full_table_df: pd.DataFrame,
    min_len: int = 0,
    min_score: int = 0,
    drop_missing: bool = True,
    fragment_mode: FragmentMode = FragmentMode.SKIP,
    duplicate_mode: DuplicateMode = DuplicateMode.SKIP,
):
    """Filter a complete BUSCO results data frame based on a set of criteria.

    Args:
        full_table_df (pandas.DataFrame): The complete BUSCO results.
        min_len (int): Minimum sequence length to keep.
        min_score (int): Minimum BUSCO score to keep.
        drop_missing (bool): If True, remove missing USCOs.
        fragment_mode (FragmentMode): How to handle Fragmented USCOs.
        duplicate_mode (DuplicateMode): How to handle Duplicated USCOs.

    Returns:
        pandas.DataFrame: The filtered BUSCO results as a Pandas DataFrame.
    """
    # Step 1: Filter by length and score
    df = full_table_df[
        (full_table_df["length"] > min_len) & (full_table_df["score"] > min_score)
    ]

    # Step 2: Apply modes
    frag_df = _filter_by_mode(df, "Fragmented", fragment_mode)
    dup_df = _filter_by_mode(df, "Duplicated", duplicate_mode)
    categories = {"Complete"} if drop_missing else {"Complete", "Missing"}
    complete_missing_df = df[df["status"].isin(categories)]

    # Step 3: Combine and return
    return pd.concat([complete_missing_df, frag_df, dup_df]).reset_index(drop=True)


def _get_corresponding_files(
    full_table_df: pd.DataFrame,
    busco_sequences_dir: Path,
    seq_type: SequenceType = SequenceType.NUCLEOTIDE,
    nucl_ext: "str" = ".fna",
    prot_ext: "str" = ".faa",
) -> list[Path]:
    """Returns a list of absolute file paths corresponding to each marker found in the
    provided data frame.

    Args:
        full_table_df (pandas.DataFrame): The complete BUSCO results.
        busco_results_dir (str): Full path to the folder containing one analysis run,
            including the name of the folder given with the flag `-o`/`--out`
        seq_type (str): Name of the sequence type to extract. Must be `nucleotide` or
            `protein`
        nucl_ext (str): Extension used for nucleotide FASTA files, including the dot.
        prot_ext (str): Extension used for amino acid FASTA files, including the dot.

    Returns:
        list[Path]: List of absolute file paths where each marker is stored.
    """
    fasta_ext = prot_ext if seq_type.is_protein() else nucl_ext

    status_location = {
        "Complete": "single_copy_busco_sequences",
        "Duplicated": "multi_copy_busco_sequences",
        "Fragmented": "fragmented_busco_sequences",
    }

    paths = full_table_df.apply(
        lambda row: Path(busco_sequences_dir)
        / status_location[row["status"]]
        / f"{row['busco_id']}{fasta_ext}",
        axis=1,
    ).tolist()

    return paths


def _normalize_fasta_header(
    header: str,
    species_tag: str = None,
    species_separator: str = "|",
    replacement_char: str = "_",
    allowed_chars: str = r"A-Za-z0-9_.\-",
) -> str:
    """
    Normalize a FASTA header and prepend with a species tag.

    Args:
        header (str): FASTA header (with leading '>').
        species_tag (str): Tag to prepend
        species_separator (str): Separator between species and header (default: '|').
        replacement_char (str): Char to replace disallowed chars and spaces.
        allowed_chars (str): Regex class of allowed characters.

    Returns:
        str: Normalized header.
    """
    # remove leading '>'
    token = header[1:].strip()

    # remove species_separator from list of allowed characters
    allowed_chars = allowed_chars.replace(species_separator, "")

    # replace disallowed characters with replacement character
    token = re.sub(f"[^{allowed_chars}]", replacement_char, token)

    # prepend species tag if provided
    if species_tag:
        token = f">{species_tag}{species_separator}{token}"

    return f">{token}\n"


@overload
def _append_uscos(
    usco_dir: GenesDirectoryFormat,
    usco_paths: list[Path],
    seq_type: SequenceType.NUCLEOTIDE,
    species_tag: str | None = None,
    species_separator: str = "|",
) -> GenesDirectoryFormat: ...


@overload
def _append_uscos(
    usco_dir: ProteinsDirectoryFormat,
    usco_paths: list[Path],
    seq_type: SequenceType.PROTEIN,
    species_tag: str | None = None,
    species_separator: str = "|",
) -> ProteinsDirectoryFormat: ...


def _append_uscos(
    usco_dir: Union[GenesDirectoryFormat, ProteinsDirectoryFormat],
    usco_paths: list[Path],
    seq_type: SequenceType = SequenceType.NUCLEOTIDE,
    species_tag: str | None = None,
    species_separator: str = "|",
    replacement_char: str = "_",
    allowed_chars: str = r"A-Za-z0-9_.\-",
) -> Union[GenesDirectoryFormat, ProteinsDirectoryFormat]:
    """Append a collection of Universal Single-Copy Ortholog (USCO) sequence files to a
    QIIME 2 directory format artifact.

    Each input FASTA file is expected to contain USCO sequences from a single MAG. If a
    FASTA file with that name already exists within the USCO directory, the sequence(s)
    are appended to the end of that file. Otherwise, a new FASTA file is created.

    FASTA headers are modified to include a species tag if provided:

        >{species_tag}{species_separator}{original_description}

    Args:
        usco_dir: Existing USCO directory.
        usco_paths (list[Path]): Paths to FASTA files containing USCO sequences.
        seq_type (SequenceType, optional): The type of sequences contained in
            `usco_paths`. Defaults to `SequenceType.NUCLEOTIDE`.
        species_tag (str): Optional species identifier to prefix FASTA headers.
        species_separator (str): Separator between species tag and sequence description.
            Defaults to "|".
        replacement_char (str): Char to replace disallowed chars and spaces.
        allowed_chars (str): Regex class of allowed characters.

    Returns:
        GenesDirectoryFormat | ProteinsDirectoryFormat:
            A directory format wrapping the provided USCO sequences, typed according
            to `seq_type`.
    """
    if seq_type.is_protein():
        usco_dir = ProteinsDirectoryFormat()
    else:
        usco_dir = GenesDirectoryFormat()

    for source_fp in usco_paths:
        destination_fp = Path(str(usco_dir)) / f"{source_fp.stem}.fasta"

        mode = "a" if destination_fp.exists() else "w"
        with open(source_fp, "r") as src, open(destination_fp, mode) as dst:
            for line in src:
                if line.startswith(">"):
                    line = _normalize_fasta_header(
                        line,
                        species_tag=species_tag,
                        species_separator=species_separator,
                        replacement_char=replacement_char,
                        allowed_chars=allowed_chars,
                    )
                dst.write(line)

    return usco_dir


def _extract_uscos(
    busco_results_dir: Path,
    lineage_dataset: str,
    seq_type: SequenceType = SequenceType.NUCLEOTIDE,
    min_len: int = 0,
    min_score: int = 0,
    fragment_mode: FragmentMode = FragmentMode.SKIP,
    duplicate_mode: DuplicateMode = DuplicateMode.SKIP,
) -> list[Path]:
    """Extract USCO sequences from the given BUSCO results folder.

    Args:
        busco_results_dir (str): Full path to the folder containing one analysis run,
            including the name of the folder given with the flag `-o`/`--out`.
        lineage_dataset (str): Name of the lineage used for this analysis run.
        seq_type (str): Name of the sequence type to extract. Must be `nucleotide` or
            `protein`.
        min_len (int): Minimum sequence length to keep.
        min_score (int): Minimum BUSCO score to keep.
        drop_missing (bool): If True, remove missing USCOs
        fragment_mode (FragmentMode): How to handle Fragmented USCOs.
        duplicate_mode (DuplicateMode): How to handle Duplicated USCOs.

    Returns:
        list[Path]: List of absolute file paths where each marker is stored.
    """
    if not busco_results_dir.is_dir():
        raise NotADirectoryError(
            "BUSCO results directory does not exist or is not a directory: "
            f"{busco_results_dir}"
        )

    analysis_run_dir = busco_results_dir / f"run_{lineage_dataset}"
    busco_sequences_dir = analysis_run_dir / "busco_sequences"
    full_table_fp = analysis_run_dir / "full_table.tsv"

    return (
        _parse_full_table_file(full_table_fp)
        .pipe(
            _filter_full_table_df,
            min_len=min_len,
            min_score=min_score,
            drop_missing=True,
            fragment_mode=FragmentMode.SKIP,
            duplicate_mode=DuplicateMode.SKIP,
        )
        .pipe(
            _get_corresponding_files,
            busco_sequences_dir=busco_sequences_dir,
            seq_type=seq_type,
        )
    )


def _get_busco_results_dir(
    output_path: str,
    output_folder_name: str,
    mag_file_path: str,
) -> Path:
    """Constructs the path to the corresponding BUSCO output folder based on the given
    parameters.

    Args:
        output_folder_path (str): Path to the location of the output folder, excluding
            the name of the folder. Corresponds to the flag `--out_path` in BUSCO.
        output_folder_name (str): Name of the output folder for this analysis run.
            Corresponds to the flag `-o`/`--out` in BUSCO.
        mag_file_path (str): Path to the input MAG file used for this analysis run.

    Returns:
        Path: Path to the BUSCO output folder for this anlaysis run.
    """
    return Path(output_path) / output_folder_name / os.path.basename(mag_file_path)


@overload
def _extract_orthologs_busco(
    *args, seq_type: SequenceType.NUCLEOTIDE, **kwargs
) -> GenesDirectoryFormat: ...


@overload
def _extract_orthologs_busco(
    *args, seq_type: SequenceType.PROTEIN, **kwargs
) -> ProteinsDirectoryFormat: ...


def _extract_orthologs_busco(
    mags: Union[MultiMAGSequencesDirFmt, MAGSequencesDirFmt],
    db: BuscoDatabaseDirFmt,
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
    num_partitions: int = None,
) -> (GenesDirectoryFormat, ProteinsDirectoryFormat):  # type: ignore
    kwargs = {
        k: v
        for k, v in locals().items()
        if k not in ["mags", "db", "additional_metrics"]
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
            kwargs,  # kwargs may be modified inside this function
        )

    # Filter out all kwargs that are None, False or 0.0
    common_args = _process_common_input_params(
        processing_func=_parse_busco_params, params=kwargs
    )

    if isinstance(mags, MultiMAGSequencesDirFmt):
        sample_dir = mags.sample_dict()
    elif isinstance(mags, MAGSequencesDirFmt):
        sample_dir = {"feature_data": mags.feature_dict()}

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
                busco_results_dir = _get_busco_results_dir(tmp, sample_id, mag_fp)
                usco_fps_nucl, usco_fps_prot = (
                    _extract_uscos(
                        busco_results_dir=busco_results_dir,
                        lineage_dataset=lineage_dataset,
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

    return usco_nucl_dir, usco_prot_dir


def extract_orthologs_busco(
    ctx,
    mags,
    db,
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
        k: v for k, v in locals().items() if k not in ["mags", "ctx", "num_partitions"]
    }

    _extract_orthologs_busco = ctx.get_action("annotate", "_extract_orthologs_busco")
    collate_busco_sequences = ctx.get_action("annotate", "collate_busco_sequences")

    if issubclass(mags.format, MultiMAGSequencesDirFmt):
        partition_action = "partition_sample_data_mags"
    else:
        partition_action = "partition_feature_data_mags"
    partition_mags = ctx.get_action("types", partition_action)

    (partitioned_mags,) = partition_mags(mags, num_partitions)

    results = list(
        map(
            lambda mag: _extract_orthologs_busco(mag, **kwargs),
            partitioned_mags.values(),
        )
    )

    nucl_dirs, prot_dirs = zip(*results)
    nucl_dirs = list(nucl_dirs)
    prot_dirs = list(prot_dirs)

    collated_nucl_dirs, collated_prot_dirs = collate_busco_sequences(
        nucl_dirs, prot_dirs
    )

    return collated_nucl_dirs, collated_prot_dirs
