"""
AccentSense Phase 2 Runner: Curation & Speaker-Disjoint Splitting.
Run with: python curate_data.py [--csv_path PATH] [--output_dir DIR]
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.curate_dataset import curate_and_export_splits
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate Svarah dataset into 4-class regional splits.")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to Svarah metadata CSV")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Output directory for CSV splits")
    args = parser.parse_args()

    curate_and_export_splits(metadata_csv_path=args.csv_path, output_dir=args.output_dir)
