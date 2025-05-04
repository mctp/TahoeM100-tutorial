#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tahoe-100M Dataset Processing Workshop
-------------------------------------
This script walks through downloading, processing, and analyzing the Tahoe-100M
single-cell genomics dataset with extensive documentation for beginners.
"""

########################
# Step 1: Package Installation
########################

import subprocess
import sys

print("Step 1: Installing required packages...")

# Function to install packages using pip
def install_package(package):
    print(f"  Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = [
    "datasets",       # For loading HuggingFace datasets
    "scipy",          # For sparse matrix operations
    "anndata",        # For working with annotated data matrices
    "pandas",         # For data manipulation
    "pubchempy",      # For chemical data
    "matplotlib",     # For visualization
    "seaborn",        # For enhanced plots
    "tqdm"            # For progress tracking
]

# Install packages if not already installed
for package in required_packages:
    try:
        __import__(package)
        print(f"  {package} is already installed.")
    except ImportError:
        install_package(package)

print("✓ All packages installed successfully!\n")

########################
# Step 2: Import Libraries
########################

print("Step 2: Importing libraries...")

# Core data processing libraries
from datasets import load_dataset
from scipy.sparse import csr_matrix
import anndata
import pandas as pd
import pubchempy as pcp
import os 
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
from tqdm.auto import tqdm
import time

print("✓ Libraries imported successfully!\n")

########################
# Step 3: Setup Environment
########################

print("Step 3: Setting up environment...")

# Create output directory in current working directory
output_dir = os.path.join(os.getcwd(), "tahoe_workshop_output")
os.makedirs(output_dir, exist_ok=True)
print(f"  Output directory created at: {output_dir}")

# Set random seed for reproducibility
np.random.seed(42)

print("✓ Environment setup complete!\n")

########################
# Step 4: Define Helper Functions
########################

print("Step 4: Defining helper functions...")

def create_anndata_from_generator(generator, gene_vocab, sample_size=None):
    """
    Convert Tahoe-100M data into an AnnData object for single-cell analysis.
    
    Parameters:
    -----------
    generator : Iterable yielding cell data from the dataset
    gene_vocab : Dictionary mapping token IDs to gene names
    sample_size : Number of cells to process (smaller values for testing)
    
    Returns:
    --------
    AnnData object containing gene expression matrix and metadata
    """
    # Sort genes to ensure consistent column ordering
    sorted_vocab_items = sorted(gene_vocab.items())
    token_ids, gene_names = zip(*sorted_vocab_items)
    
    # Create mapping from gene IDs to matrix columns
    token_id_to_col_idx = {token_id: idx for idx, token_id in enumerate(token_ids)}
    
    # Initialize sparse matrix components
    data, indices, indptr = [], [], [0]
    obs_data = []
    
    # Process each cell with progress tracking
    print(f"  Processing {sample_size if sample_size else 'all'} cells...")
    for i, cell in enumerate(tqdm(generator, total=sample_size)):
        if sample_size is not None and i >= sample_size:
            break
            
        # Extract gene expression data
        genes = cell['genes']
        expressions = cell['expressions']
        
        # Handle metadata indicator (negative first value)
        if expressions[0] < 0:
            genes = genes[1:]
            expressions = expressions[1:]
        
        # Map genes to column indices and filter valid expressions
        col_indices = [token_id_to_col_idx[gene] for gene in genes if gene in token_id_to_col_idx]
        valid_expressions = [expr for gene, expr in zip(genes, expressions) if gene in token_id_to_col_idx]
        
        # Add to sparse matrix construction lists
        data.extend(valid_expressions)
        indices.extend(col_indices)
        indptr.append(len(data))
        
        # Extract cell metadata
        obs_entry = {k: v for k, v in cell.items() if k not in ['genes', 'expressions']}
        obs_data.append(obs_entry)
    
    # Create the sparse expression matrix
    expr_matrix = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(gene_names)))
    
    # Create observation DataFrame (cell metadata)
    obs_df = pd.DataFrame(obs_data)
    
    # Create AnnData object (industry standard for single-cell data)
    adata = anndata.AnnData(X=expr_matrix, obs=obs_df)
    
    # Set gene names as column labels
    adata.var.index = pd.Index(gene_names, name='ensembl_id')
    
    return adata

# Display data overview function
def plot_dataset_overview(adata, output_dir):
    """Create basic visualizations of the processed dataset"""
    print("  Creating dataset visualizations...")
    
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Cell counts by cell line (only if column exists)
    if 'cell_line' in adata.obs.columns:
        plt.figure(figsize=(12, 6))
        cell_counts = adata.obs['cell_line'].value_counts().sort_values(ascending=False)
        sns.barplot(x=cell_counts.index, y=cell_counts.values)
        plt.title('Number of Cells per Cell Line')
        plt.xlabel('Cell Line')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "cell_line_counts.png"), dpi=300)
        plt.close()
    else:
        print("  Skipping cell line visualization (column not found)")
    
    # Top drug treatments
    plt.figure(figsize=(12, 6))
    drug_counts = adata.obs['drug'].value_counts().head(20)
    sns.barplot(x=drug_counts.index, y=drug_counts.values)
    plt.title('Top 20 Drug Treatments')
    plt.xlabel('Drug')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "drug_counts.png"), dpi=300)
    plt.close()
    
    print(f"  Visualizations saved to {vis_dir}")

print("✓ Helper functions defined!\n")

########################
# Step 5: Load Dataset
########################

print("Step 5: Loading Tahoe-100M dataset...")
print("  This may take a while depending on your internet connection...")

# Load dataset with streaming to handle the large size efficiently
tahoe_100m_ds = load_dataset('vevotx/Tahoe-100M', streaming=True, split='train').shuffle(seed=42, buffer_size=100000)
print("  Dataset loaded in streaming mode with random shuffling")

# Load gene mapping information
print("  Loading gene metadata...")
gene_metadata = load_dataset("vevotx/Tahoe-100M", name="gene_metadata", split="train")
gene_vocab = {entry["token_id"]: entry["ensembl_id"] for entry in gene_metadata}
print(f"  Gene vocabulary created with {len(gene_vocab)} genes")

print("✓ Dataset loaded successfully!\n")

########################
# Step 6: Create AnnData Object
########################

print("Step 6: Creating AnnData object...")
print("  This step processes the expression data (will take several minutes)...")

# Adjust this number for workshop timing - smaller is faster
sample_size = 500  # Reduced for quicker workshop execution
start_time = time.time()

# Process cells into AnnData format
adata = create_anndata_from_generator(tahoe_100m_ds, gene_vocab, sample_size=sample_size)

# Report basic statistics
print(f"  Created AnnData object with {adata.n_obs} cells and {adata.n_vars} genes")
print(f"  Processing time: {int(time.time() - start_time)} seconds")

print("✓ AnnData object created successfully!\n")


########################
# Step 7: Enrich with Metadata
########################

print("Step 7: Enriching with metadata...")

# Load and merge sample metadata
print("  Loading sample metadata...")
sample_metadata = load_dataset("vevotx/Tahoe-100M","sample_metadata", split="train").to_pandas()

# Join sample information with expression data
print("  Merging sample metadata with AnnData object...")
adata.obs = pd.merge(
    adata.obs,
    sample_metadata.drop(columns=["plate"]),
    on="sample",
    suffixes=('_original', '_sample_meta')
)

# Use the most reliable drug information source
adata.obs['drug'] = adata.obs['drug_sample_meta'].combine_first(adata.obs['drug_original'])
print(f"  Sample metadata merged successfully - added {len(sample_metadata.columns)} columns")

# Load and merge drug metadata
print("  Loading drug metadata...")
drug_metadata = load_dataset("vevotx/Tahoe-100M","drug_metadata", split="train").to_pandas()

# Add drug properties to cells
print("  Merging drug metadata with AnnData object...")
adata.obs = pd.merge(
    adata.obs,
    drug_metadata.drop(columns=["canonical_smiles","pubchem_cid","moa-fine"]),
    on="drug",
    how='left'
)
print(f"  Drug metadata merged successfully - {drug_metadata.shape[0]} drugs")

# Load cell line information
print("  Loading cell line metadata...")
cell_line_metadata = load_dataset("vevotx/Tahoe-100M","cell_line_metadata", split="train").to_pandas()
print(f"  Cell line metadata loaded with {len(cell_line_metadata)} cell lines")

print("✓ Metadata enrichment complete!\n")

########################
# Step 8: Verify Data Quality
########################

print("Step 8: Verifying data quality...")

# Check for control samples (DMSO)
print("\n=== Control Sample Check ===")
print("Total cells:", adata.n_obs)
control_count = adata.obs[adata.obs['drug'] == 'DMSO_TF'].shape[0]
print(f"DMSO_TF control cells: {control_count} ({control_count/adata.n_obs:.1%} of total)")

if control_count > 0:
    print("Example control metadata:")
    print(adata.obs[adata.obs['drug'] == 'DMSO_TF'].head())

# Check expression matrix properties
print("\n=== Expression Matrix Statistics ===")
print(f"Total non-zero expressions: {adata.X.nnz}")
print(f"Sparsity: {1 - adata.X.nnz / (adata.n_obs * adata.n_vars):.4f}")
print(f"Average genes detected per cell: {adata.X.nnz / adata.n_obs:.1f}")

print("✓ Data quality verification complete!\n")

########################
# Step 9: Save Processed Data
########################

print("Step 9: Saving processed data...")

# Save metadata tables for reference
print("  Saving metadata tables...")
sample_metadata.to_csv(os.path.join(output_dir, "sample_metadata.csv"))
drug_metadata.to_csv(os.path.join(output_dir, "drug_metadata.csv")) 
cell_line_metadata.to_csv(os.path.join(output_dir, "cell_line_metadata.csv"))

# Handle complex data types before saving
problematic_columns = ['targets']
for col in problematic_columns:
    if col in adata.obs.columns:
        print(f"  Converting {col} to string representation")
        adata.obs[col] = adata.obs[col].astype(str)

# Save the final AnnData object
print("  Saving AnnData object...")
adata_file_path = os.path.join(output_dir, "tahoe_adata.h5ad")
adata.write_h5ad(adata_file_path)
print(f"  AnnData saved to: {adata_file_path}")

print("✓ All data saved successfully!\n")

########################
# Step 10: Generate Visualizations
########################

# Fix: Join cell line metadata with AnnData object
# First check if a joining column exists
if 'cell_id' in adata.obs.columns and 'cell_id' in cell_line_metadata.columns:
    print("  Merging cell line metadata with AnnData object...")
    adata.obs = pd.merge(
        adata.obs,
        cell_line_metadata,
        on="cell_id",  # Adjust this key if needed
        how='left'
    )
    print(f"  Cell line metadata merged successfully")
else:
    # As a fallback, make the visualization function more robust
    print("  Warning: Cannot merge cell line metadata - joining column not found")


print("Step 10: Generating visualizations...")
plot_dataset_overview(adata, output_dir)
print("✓ Visualizations complete!\n")

########################
# Step 11: Summary
########################

print("=" * 60)
print("WORKSHOP SUMMARY")
print("=" * 60)
print(f"• Dataset: Tahoe-100M ({adata.n_obs:,} cells, {adata.n_vars:,} genes)")
# print(f"• Cell types: {adata.obs['cell_line'].nunique():,} unique cell lines")
print(f"• Drug treatments: {adata.obs['drug'].nunique():,} unique compounds")
print(f"• Output location: {output_dir}")
print("=" * 60)
print("\nYour processed dataset is ready for analysis!")
print("Next steps could include:")
print("1. Dimensionality reduction (PCA, UMAP)")
print("2. Clustering analysis")
print("3. Differential expression testing")
print("4. Drug response modeling")
print("=" * 60)
