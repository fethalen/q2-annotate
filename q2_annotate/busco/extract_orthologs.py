import os
import re
from enum import Enum
from pathlib import Path
from typing import Union, overload

import numpy as np
import pandas as pd
from q2_types.genome_data import GenesDirectoryFormat, ProteinsDirectoryFormat
import skbio


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


class CaseMode(str, Enum):
    LOWER = "lower"
    UPPER = "upper"
    PRESERVE = "preserve"


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


def _filter_by_mode(df: pd.DataFrame, mode: Enum) -> pd.DataFrame:
    """Helper to filter a subset of BUSCO rows by mode."""
    if isinstance(mode, FragmentMode):
        status = "Fragmented"
    elif isinstance(mode, DuplicateMode):
        status = "Duplicated"
    else:
        raise ValueError(f"Unexpected mode type: {mode}")

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
    subset = subset.reset_index(drop=True)
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
    # Ensures that missing entries are not filtered out at this step
    mask = (full_table_df["length"] > min_len) & (full_table_df["score"] > min_score)
    df = full_table_df[mask | (full_table_df["status"] == "Missing")]

    frag_df = _filter_by_mode(df, fragment_mode)
    dup_df = _filter_by_mode(df, duplicate_mode)
    categories = {"Complete"} if drop_missing else {"Complete", "Missing"}
    complete_missing_df = df[df["status"].isin(categories)]

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
        seq_type (SequenceType): Name of the sequence type to extract. Must be either
            `nucleotide` or `protein`. Default: `SequenceType.NUCLEOTIDE`.
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

    if full_table_df.empty:
        return []

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
        header (str): FASTA header (without leading '>').
        species_tag (str): Tag to prepend
        species_separator (str): Separator between species and header (default: '|').
        replacement_char (str): Char to replace disallowed chars and spaces.
        allowed_chars (str): Regex class of allowed characters.

    Returns:
        str: Normalized header.
    """
    # remove species_separator from list of allowed characters
    allowed_chars = allowed_chars.replace(species_separator, "")

    # replace disallowed characters with replacement character
    token = re.sub(f"[^{allowed_chars}]", replacement_char, header)

    # prepend species tag if provided
    if species_tag:
        token = f"{species_tag}{species_separator}{token}"

    return f"{token}\n"


@overload
def _append_uscos(
    usco_dir: GenesDirectoryFormat,
    usco_paths: list[Path],
    seq_type: SequenceType.NUCLEOTIDE,
    species_tag: str | None = None,
    species_separator: str = "|",
    replacement_char: str = "_",
    allowed_chars: str = r"A-Za-z0-9_.\-",
    wrap_column: int = 0,
    case_mode: CaseMode = CaseMode.PRESERVE,
) -> GenesDirectoryFormat: ...


@overload
def _append_uscos(
    usco_dir: ProteinsDirectoryFormat,
    usco_paths: list[Path],
    seq_type: SequenceType.PROTEIN,
    species_tag: str | None = None,
    species_separator: str = "|",
    replacement_char: str = "_",
    allowed_chars: str = r"A-Za-z0-9_.\-",
    wrap_column: int = 0,
    case_mode: CaseMode = CaseMode.PRESERVE,
) -> ProteinsDirectoryFormat: ...


def _append_uscos(
    usco_dir: Union[GenesDirectoryFormat, ProteinsDirectoryFormat],
    usco_paths: list[Path],
    seq_type: SequenceType = SequenceType.NUCLEOTIDE,
    species_tag: str | None = None,
    species_separator: str = "|",
    replacement_char: str = "_",
    allowed_chars: str = r"A-Za-z0-9_.\-",
    wrap_column: int = 0,
    case_mode: CaseMode = CaseMode.PRESERVE,
) -> Union[GenesDirectoryFormat, ProteinsDirectoryFormat]:
    """Append a collection of Universal Single-Copy Ortholog (USCO) sequence files to a
    QIIME 2 directory format artifact.

    Each input FASTA file is expected to contain USCO sequences from a single MAG. If a
    FASTA file with that name already exists within the USCO directory, the sequence(s)
    are appended to the end of that file. Otherwise, a new FASTA file is created.

    FASTA headers are modified to include a species tag if provided:

        >{species_tag}{species_separator}{original_description}

    Sequences are either wrapped at the provided wrap_column or, if the wrap_column is
    set to 0, each sequence is written to a single line only.

    Args:
        usco_dir: Existing USCO directory.
        usco_paths (list[Path]): Paths to FASTA files containing USCO sequences.
        seq_type (SequenceType): The type of sequences contained in
            `usco_paths`. Defaults to `SequenceType.NUCLEOTIDE`.
        species_tag (str): Optional species identifier to prefix FASTA headers.
        species_separator (str): Separator between species tag and sequence description.
            Defaults to "|".
        replacement_char (str): Char to replace disallowed chars and spaces. Default to
            "_".
        allowed_chars (str): Regex class of allowed characters.
        wrap_column (int): Wrap sequences at the provided number. If 0, output
            sequences sequentially (i.e., one sequence per line).
        case_mode (CaseMode): Determines sequence casing. Must be one of the following:
            "lower", "upper", or "preserve". Default: "preserve".

    Returns:
        GenesDirectoryFormat | ProteinsDirectoryFormat:
            A directory format wrapping the provided USCO sequences, typed according
            to `seq_type`.
    """
    max_width = None if wrap_column == 0 else wrap_column

    for source_fp in usco_paths:
        destination_fp = Path(str(usco_dir)) / f"{source_fp.stem}.fasta"

        sequences = skbio.read(
            str(source_fp),
            format="fasta",
            constructor=skbio.DNA if seq_type.is_nucleotide() else skbio.Protein,
        )

        mode = "a" if destination_fp.exists() else "w"

        with open(destination_fp, mode) as out_f:
            for seq in sequences:
                full_header = seq.metadata["id"]
                if seq.metadata["description"]:
                    full_header += " " + seq.metadata["description"]

                header = _normalize_fasta_header(
                    full_header,
                    species_tag=species_tag,
                    species_separator=species_separator,
                    replacement_char=replacement_char,
                    allowed_chars=allowed_chars,
                )

                seq.metadata["id"] = header
                seq.metadata["description"] = ""

                if case_mode == CaseMode.LOWER:
                    lowercase = np.ones(len(seq), dtype=bool)
                elif case_mode == CaseMode.UPPER:
                    lowercase = None
                elif case_mode == CaseMode.PRESERVE:
                    lowercase = np.array([c.islower() for c in str(seq)])
                else:
                    raise ValueError(f"Invalid case_mode: {case_mode}")

                skbio.write(
                    seq,
                    format="fasta",
                    into=out_f,
                    max_width=max_width,
                    lowercase=lowercase,
                )

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
            fragment_mode=fragment_mode,
            duplicate_mode=duplicate_mode,
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
