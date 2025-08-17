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
import importlib.util
import subprocess
from pathlib import Path

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

def run_etl_pipeline(project_root: Path):
    """Run the ETL pipeline to transform raw data."""
    print("Transformed data not found. Running ETL pipeline...")

    try:
        # Change directory to etl to ensure config is found
        etl_dir = project_root / "etl"
        script_path = etl_dir / "etl_pipeline.py"

        # Run the ETL script in its own directory context
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(etl_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"ERROR: ETL pipeline failed: {result.stderr}")
            raise Exception(result.stderr)
        else:
            print("ETL pipeline completed successfully!")
            if result.stdout.strip():
                print(result.stdout)
    except Exception as e:
        print(f"ERROR: ETL pipeline failed: {str(e)}")
        raise

def run_figure_script(script_path, script_name):
    """Run a figure generation script using subprocess to avoid import issues."""
    print(f"Generating {script_name}...")
    try:
        # Use subprocess to run the script in its own directory to avoid import issues
        script_dir = script_path.parent
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"ERROR: Failed to generate {script_name}:")
            print(result.stderr)
        else:
            print(f"Successfully generated {script_name}")
            if result.stdout.strip():
                print(result.stdout)
    except Exception as e:
        print(f"ERROR: Failed to generate {script_name}: {str(e)}")
        print("Continuing with next figure...")

def generate_figures(project_root):
    """Generate all four figures for the project."""
    figure_scripts = [
        ("Figure 1 (Platform)", project_root / "figures" / "figure_1_platform" / "main_fig1.py"),
        ("Figure 2 (Theory)", project_root / "figures" / "figure_2_theory" / "main_figure_2.py"),
        ("Figure 3 (Experiment)", project_root / "figures" / "figure_3_experiment" / "main_figure_3.py"),
        ("Figure 4 (Metrics)", project_root / "figures" / "figure_4_metrics" / "main_figure_4.py")
    ]

    for fig_name, script_path in figure_scripts:
        run_figure_script(script_path, fig_name)

def main():
    """Main entry point for the EP-TPD Unification project."""
    print("="*80)
    print("EP-TPD Unification Project")
    print("="*80)

    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
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
            run_etl_pipeline(project_root)
        except Exception:
            print("Exiting due to ETL pipeline failure.")
            return 1
    else:
        print("Transformed data already exists. Skipping ETL pipeline.")

    # Step 3: Generate all figures
    print("\nGenerating figures...")
    generate_figures(project_root)

    print("\n"+"="*80)
    print("EP-TPD Unification Project Completed Successfully!")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
