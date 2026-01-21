from pathlib import Path

def create_project_structure(root_name="muFTIR_microplastic_OSR"):
    root = Path(root_name)

    directories = [
        # Data layer
        root / "data" / "raw" / "hdf5_original",
        root / "data" / "raw" / "metadata",
        root / "data" / "registry",
        root / "data" / "standardized" / "spectra_npz",
        root / "data" / "standardized" / "logs",
        root / "data" / "splits" / "closed_set",
        root / "data" / "splits" / "open_set",
        root / "data" / "splits" / "stress_tests",

        # Source code
        root / "src" / "audit",
        root / "src" / "preprocessing",
        root / "src" / "models",
        root / "src" / "evaluation",
        root / "src" / "utils",

        # Experiments & checks
        root / "notebooks",
        root / "reports" / "figures",
        root / "reports" / "tables",

        # Configs & docs
        root / "configs",
        root / "tests",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

    # Create placeholder files
    placeholder_files = [
        root / "README.md",
        root / "configs" / "paths.yaml",
        root / "configs" / "preprocessing.yaml",
        root / "src" / "audit" / "__init__.py",
        root / "src" / "preprocessing" / "__init__.py",
        root / "src" / "models" / "__init__.py",
        root / "src" / "evaluation" / "__init__.py",
        root / "src" / "utils" / "__init__.py",
    ]

    for file in placeholder_files:
        if not file.exists():
            file.touch()
            print(f"Created file: {file}")

    print("\n✅ Project structure successfully initialized.")

if __name__ == "__main__":
    create_project_structure()