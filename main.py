#!/usr/bin/env python
"""
Main entry point for the EP-TPD Unification project.

This script:
1. Verifies all required raw databases are present
2. Checks if transformed data exists, running the ETL pipeline if needed
3. Generates all four figures for the project
"""

import os
import sys
from pathlib import Path
from typing import List

# Check for required databases
def check_raw_databases(data_dir: Path) -> bool:
    """
    Check if all required raw databases are present.

    Returns:
        bool: True if all databases are present, False otherwise
    """
    required_dbs = [
        "ep_tpd_experiment_data.db",
        "fig1_raw_data.db"
    ]

    missing_dbs = []
    for db in required_dbs:
        if not (data_dir / db).exists():
            missing_dbs.append(db)

    if missing_dbs:
        print(f"ERROR: Missing raw databases: {', '.join(missing_dbs)}")
        print(f"Please ensure all raw databases are in the {data_dir} directory.")
        return False

    return True

def check_transformed_data(data_dir: Path) -> bool:
    """
    Check if the transformed data database exists.

    Returns:
        bool: True if transformed data exists, False otherwise
    """
    return (data_dir / "ep_tpd_transformed_data.db").exists()

def run_etl_pipeline():
    """Run the ETL pipeline to transform raw data."""
    print("Transformed data not found. Running ETL pipeline...")

    try:
        # Import here to avoid circular imports
        from etl.etl_pipeline import main as etl_main
        etl_main()
        print("ETL pipeline completed successfully!")
    except Exception as e:
        print(f"ERROR: ETL pipeline failed: {str(e)}")
        raise

def generate_figures():
    """Generate all four figures for the project."""
    figure_modules = [
        ("Figure 1 (Platform)", "figures.figure_1_platform.main_fig1"),
        ("Figure 2 (Theory)", "figures.figure_2_theory.main_figure_2"),
        ("Figure 3 (Experiment)", "figures.figure_3_experiment.main_figure_3"),
        ("Figure 4 (Metrics)", "figures.figure_4_metrics.main_figure_4")
    ]

    for fig_name, module_path in figure_modules:
        print(f"Generating {fig_name}...")
        try:
            module = __import__(module_path, fromlist=["main"])
            if hasattr(module, "main"):
                module.main()
            else:
                print(f"WARNING: No main function found in {module_path}")
        except Exception as e:
            print(f"ERROR: Failed to generate {fig_name}: {str(e)}")
            print("Continuing with next figure...")

def main():
    """Main entry point for the EP-TPD Unification project."""
    print("="*80)
    print("EP-TPD Unification Project")
    print("="*80)

    # Get the project root directory
    project_root = Path(__file__).parent
    data_dir = project_root / "data"

    # Step 1: Check if all raw databases are present
    print("\nChecking for required raw databases...")
    if not check_raw_databases(data_dir):
        print("Exiting due to missing raw databases.")
        return 1
    print("All required raw databases found!")

    # Step 2: Check if transformed data exists
    print("\nChecking for transformed data...")
    if not check_transformed_data(data_dir):
        try:
            run_etl_pipeline()
        except Exception:
            print("Exiting due to ETL pipeline failure.")
            return 1
    else:
        print("Transformed data already exists. Skipping ETL pipeline.")

    # Step 3: Generate all figures
    print("\nGenerating figures...")
    generate_figures()

    print("\n"+"="*80)
    print("EP-TPD Unification Project Completed Successfully!")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
