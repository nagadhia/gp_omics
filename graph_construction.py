import os
import pickle
import time
import traceback
import warnings
from collections import defaultdict
import random

import pandas as pd
import numpy as np
import networkx as nx # Ensure networkx is imported
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy import stats
from joblib import Parallel, delayed
try:
    from tqdm.auto import tqdm as tqdm_func
except ImportError:
    def tqdm_func(iterable, *args, **kwargs):
        return iterable


# Utility Functions ---
def safe_filename(name):
    """Creates a safe string for paths/filenames."""
    return str(name).replace('/', '-').replace('\\', '-').replace(':', '-').replace(' ', '_')

# Pre Processing ---

def process_data(otu_file_path, metadata_file_path, output_dir,
                 sample_id='Sample', condition_id='Study.Group'):

    print(f" Reading OTU table at: {otu_file_path}")
    print(f" Reading metadata at: {metadata_file_path}")

    processed_data_dir = os.path.join(output_dir, "01_Processed_Data")
    os.makedirs(processed_data_dir, exist_ok=True)

    sep_otu = '\t' if otu_file_path.lower().endswith('.tsv') else ','
    sep_meta = '\t' if metadata_file_path.lower().endswith('.tsv') else ','

    try:
        otu_table = pd.read_csv(otu_file_path, index_col=0, sep=sep_otu)
        print(f"  Initial OTU table shape: {otu_table.shape}")
        metadata = pd.read_csv(metadata_file_path, sep=sep_meta)
        print(f"  Initial Metadata shape: {metadata.shape}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error reading input files: {e}. Please check paths.")
    except Exception as e:
        raise ValueError(f"Error parsing input files: {e}")

    if sample_id not in metadata.columns:
        raise ValueError(f"Sample ID column '{sample_id}' not found in metadata. Available columns: {metadata.columns.tolist()}")
    if condition_id not in metadata.columns:
        raise ValueError(f"Condition ID column '{condition_id}' not found in metadata. Available columns: {metadata.columns.tolist()}")
    print(f"  Using Sample ID column: '{sample_id}'")
    print(f"  Using Condition ID column: '{condition_id}'")
    if not metadata[sample_id].is_unique:
        num_duplicates = metadata[sample_id].duplicated().sum()
        print(f"  Warning: Sample ID column '{sample_id}' contains {num_duplicates} duplicate values. Keeping first occurrence of each.")
        metadata = metadata.drop_duplicates(subset=[sample_id], keep='first')
        print(f"  Metadata shape after dropping duplicates: {metadata.shape}")
    metadata.set_index(sample_id, inplace=True)

  
    otu_table.index = otu_table.index.astype(str)
    metadata.index = metadata.index.astype(str)
    common_samples = otu_table.index.intersection(metadata.index)
    print(f"  Found {len(common_samples)} common samples between OTU table and metadata.")
    if len(common_samples) == 0:
        raise ValueError("No common samples found. Check Sample ID matching and formatting.")
    otu_table = otu_table.loc[common_samples]
    metadata = metadata.loc[common_samples]
    print(f"  Aligned feature table shape: {otu_table.shape}")
    print(f"  Aligned metadata shape: {metadata.shape}")

   
    try:
        otu_table = otu_table.astype(float)
    except ValueError as e:
        non_numeric_cols = otu_table.apply(lambda s: pd.to_numeric(s, errors='coerce').isna().any())
        problem_cols = non_numeric_cols[non_numeric_cols].index.tolist()
        raise ValueError(f"OTU table contains non-numeric values. Problematic columns might include: {problem_cols}. Original error: {e}")

    initial_otus = otu_table.shape[1]
    otu_table = otu_table.loc[:, (otu_table > 0).any(axis=0)] # Remove OTUs with all zeros
    otus_removed = initial_otus - otu_table.shape[1]
    if otus_removed > 0:
        print(f"  Removed {otus_removed} OTUs that were all zero across samples.")

    sample_sums = otu_table.sum(axis=1)
    valid_samples_mask = sample_sums > 1e-9 # Find samples with non-zero sum
    if (~valid_samples_mask).any():
        print(f"  Warning: Removing {(~valid_samples_mask).sum()} samples with zero total abundance before normalization.")
        otu_table = otu_table.loc[valid_samples_mask]
        metadata = metadata.loc[valid_samples_mask]
        sample_sums = sample_sums.loc[valid_samples_mask]

    if sample_sums.empty:
        print("  Warning: No samples remaining after filtering zero-sum samples. Resulting table will be empty.")
        final_otu_table = pd.DataFrame(columns=otu_table.columns, index=otu_table.index)
        otu_table_grouped = None
    else:
        otu_table = otu_table.div(sample_sums, axis=0) # Normalize to relative abundance
        otu_table = otu_table.fillna(0)
        final_otu_table = otu_table
        print(f"  Normalized OTU table to relative abundances. Final shape: {final_otu_table.shape}")
        # Grouping (optional, not saved by default)
        otu_table_grouped = None
        if condition_id in metadata.columns:
             try:
                if not metadata.empty:
                     otu_table_grouped = final_otu_table.groupby(metadata[condition_id]).mean()
             except Exception as group_e:
                 print(f"  Warning: Could not group OTU table by condition '{condition_id}': {group_e}")

    # Save the processed data
    try:
        ft_path = os.path.join(processed_data_dir, "feature_table_normalized.csv")
        meta_path = os.path.join(processed_data_dir, "metadata_aligned.csv")
        final_otu_table.to_csv(ft_path)
        metadata.to_csv(meta_path)
        print(f"  Saved normalized feature table to: {ft_path}")
        print(f"  Saved aligned metadata to: {meta_path}")
    except Exception as e:
        print(f"  Warning: Could not save processed data files: {e}")

    print(" Data processing function finished.")
    return final_otu_table, metadata

def compute_sample_weights(sample_rel_otu):
    """Calculates pairwise weights within a single sample based on relative abundance."""
    sample_rel_otu = np.asarray(sample_rel_otu)
    n_species = len(sample_rel_otu)

    if n_species < 2:
        return np.zeros((n_species, n_species)), np.zeros((n_species, n_species))

    binary_sample = (sample_rel_otu > 1e-9).astype(int)
    sample_binary_matrix = np.outer(binary_sample, binary_sample)
    sample_matrix = np.tile(sample_rel_otu, (n_species, 1))

    original_array = sample_rel_otu # Diagonal elements = abundances
    original_array_safe = original_array + np.finfo(float).eps
    non_zero_mask = original_array > 1e-9

    inverted_non_zero_elements = np.zeros_like(original_array, dtype=float)
    inverted_non_zero_elements = np.divide(1.0, original_array_safe, where=non_zero_mask, out=inverted_non_zero_elements)
    inv_diag = np.diag(inverted_non_zero_elements)

    ratios = np.matmul(inv_diag, sample_matrix) # R_ij = abundance_j / abundance_i

    weights = np.zeros_like(ratios)
    non_diagonal_mask = ~np.eye(n_species, dtype=bool)
    ratios_safe = ratios + np.finfo(float).eps
    valid_ratios_mask = (np.abs(ratios) > 1e-9) & non_diagonal_mask

    # Weight calculation W_ij = 2 / R_ij = 2 * abundance_i / abundance_j
    weights = np.divide(2.0, ratios_safe, where=valid_ratios_mask, out=weights)

    # Symmetrization using upper triangle: W'_ij = W_ij, W'_ji = W_ij for i < j
    weights_new = np.triu(weights, k=1)
    weights_new = weights_new + weights_new.T
    weights_new[~np.isfinite(weights_new)] = 0 # Handle potential infinities/NaNs

    return weights_new, sample_binary_matrix

def compute_all_weights(raw_data):
    """Computes an average weight matrix across a set of samples."""
    raw_data = np.asarray(raw_data)
    if raw_data.ndim != 2 or raw_data.shape[0] == 0 or raw_data.shape[1] == 0:
         print("Warning: compute_all_weights received empty or invalid data.")
         num_species = raw_data.shape[1] if raw_data.ndim == 2 else 0
         return np.zeros((num_species, num_species)), np.zeros((num_species, num_species))

    # Normalize rows (samples) 
    relative_raw = normalize(raw_data, axis=1, norm='l1')
    num_samples, num_species = relative_raw.shape
    if num_species == 0: # Check after normalization
        return np.zeros((0,0)), np.zeros((0,0))

    combined_weights = np.zeros((num_species, num_species))
    cooc_matrix = np.zeros((num_species, num_species)) # Sum of co-occurrences

    for i in range(num_samples):
        sample = relative_raw[i, :]
        if np.sum(sample) < 1e-9: # Skip empty samples post-normalization
            continue
        if len(sample) < 2: # Skip if too few species
             continue

        w, cooc = compute_sample_weights(sample)
        # Add only if shapes match
        if w.shape == combined_weights.shape and cooc.shape == cooc_matrix.shape:
             combined_weights += w
             cooc_matrix += cooc
        else:
            print(f"Warning: Shape mismatch in compute_all_weights sample {i}. Skipping.")

    # Average weights: Divide sum of weights by number of times species co-occurred
    final_matrix = np.divide(combined_weights, cooc_matrix, where=cooc_matrix!=0, out=np.zeros_like(combined_weights))
    final_matrix[~np.isfinite(final_matrix)] = 0 # Handle potential NaNs/Infs

    return final_matrix, cooc_matrix

def create_individual_unfiltered_graph(sample_data_series):
    """Creates an unfiltered NetworkX graph and weights matrix for a single sample."""
    G = nx.Graph()
    numeric_sample_data = pd.to_numeric(sample_data_series, errors='coerce').fillna(0)
    nonzero_species = numeric_sample_data[numeric_sample_data > 1e-9]

    species_list = nonzero_species.index.tolist()
    num_species = len(species_list)
    weights_matrix = np.zeros((num_species, num_species)) 

    if num_species == 0:
        return G, weights_matrix # Return empty graph and matrix

    # Add nodes first
    present_species_map = {} # Map species name to index in nonzero_species
    for idx, (species, abundance) in enumerate(nonzero_species.items()):
        if pd.notna(species):
             G.add_node(species, relab=abundance)
             present_species_map[species] = idx
        else:
             print("Warning: Skipping node with invalid name (NaN?)")

    if num_species < 2:
        return G, weights_matrix

    # Calculate weights only if more than one species
    abundance_array = nonzero_species.to_numpy()
    try:
        weights_matrix, _ = compute_sample_weights(abundance_array)
        if weights_matrix.shape != (num_species, num_species):
             print(f" Warning: Weight matrix shape mismatch. Expected ({num_species}, {num_species}), got {weights_matrix.shape}.")
             weights_matrix = np.zeros((num_species, num_species)) # Reset weights matrix
             return G, weights_matrix

    except Exception as e:
         print(f" Error calculating sample weights: {e}")
         return G, weights_matrix

    # Add edges to the graph based on the calculated weights
    for i in range(num_species):
        for j in range(i + 1, num_species):
            weight = weights_matrix[i, j]
            node_i = species_list[i]
            node_j = species_list[j]

            # Add edge only if nodes exist in the graph and weight is non-zero finite
            if G.has_node(node_i) and G.has_node(node_j):
                if np.isfinite(weight) and abs(weight) > 1e-9: # Using tolerance for non-zero
                    G.add_edge(node_i, node_j, weight=weight)

    return G, weights_matrix

def create_and_save_unfiltered_sample_results(feature_table, output_dir):
    """Creates and saves unfiltered graphml graphs and weight matrices for each sample."""
    print(f"\n--- Generating Unfiltered Sample Graphs (GraphML) & Weights ---")
    unfiltered_base_dir = os.path.join(output_dir, "02_Unfiltered_Sample_Results")
    os.makedirs(unfiltered_base_dir, exist_ok=True)
    print(f" Unfiltered results will be saved in: '{unfiltered_base_dir}'")
    num_samples = len(feature_table)
    start_time = time.time()
    samples_processed = 0
    errors_encountered = 0

    sample_iterator = tqdm_func(feature_table.iterrows(), total=num_samples, desc=" Unfiltered Samples", leave=False, ncols=100)

    for i, (sample_id, sample_data) in enumerate(sample_iterator):
        safe_sample_id = safe_filename(sample_id)
        sample_output_dir = os.path.join(unfiltered_base_dir, safe_sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)

        try:
            # Create unfiltered graph and get weights matrix
            G_unfiltered, weights_matrix = create_individual_unfiltered_graph(sample_data)

            # Save graph as GraphML if it has nodes
            if G_unfiltered.number_of_nodes() > 0:
                graph_path = os.path.join(sample_output_dir, "graph.graphml") 
                try:
                    nx.write_graphml(G_unfiltered, graph_path) 
                except Exception as save_err:
                     print(f"\n ERROR saving unfiltered graph {graph_path}: {save_err}")
            
            if weights_matrix.size > 0 : # Check if matrix is not empty (e.g. shape (0,0))
                weights_path = os.path.join(sample_output_dir, "weights.csv")
                try:
                    np.savetxt(weights_path, weights_matrix, delimiter=",")
                except Exception as save_err:
                    print(f"\n ERROR saving weights matrix {weights_path}: {save_err}")

            samples_processed += 1
        except Exception as e:
            errors_encountered += 1
            print(f"\n ERROR processing unfiltered results for sample {sample_id}: {e}")
            traceback.print_exc() # Print detailed traceback for debugging

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Finished Unfiltered Sample Results Generation ---")
    print(f"  Total time: {duration:.2f} seconds")
    print(f"  Samples processed: {samples_processed}, Errors: {errors_encountered}")

def save_simple_filtered_individual_graphs(feature_table, output_dir, edge_threshold):
    """Loads unfiltered weights, applies simple threshold, saves filtered graphml graph."""
    print(f"\n--- Generating Simple Filtered Individual Sample Graphs (GraphML) (Threshold > {edge_threshold}) ---")
    unfiltered_base_dir = os.path.join(output_dir, "02_Unfiltered_Sample_Results")
    simple_filtered_base_dir = os.path.join(output_dir, f"03_Simple_Filtering_(Threshold_{edge_threshold})", "Individual_Samples")
    os.makedirs(simple_filtered_base_dir, exist_ok=True)
    print(f" Filtered graphs will be saved in: '{simple_filtered_base_dir}'")
    num_samples = len(feature_table)
    start_time = time.time()
    samples_processed = 0
    errors_encountered = 0
    graphs_saved = 0

    sample_iterator = tqdm_func(feature_table.iterrows(), total=num_samples, desc=" Simple Filtering Samples", leave=False, ncols=100)

    for i, (sample_id, sample_data) in enumerate(sample_iterator):
        safe_sample_id = safe_filename(sample_id)
        sample_unfiltered_dir = os.path.join(unfiltered_base_dir, safe_sample_id)
        weights_path = os.path.join(sample_unfiltered_dir, "weights.csv")

        sample_output_dir = os.path.join(simple_filtered_base_dir, safe_sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)
        filtered_graph_path = os.path.join(sample_output_dir, "graph_filtered.graphml") 

        try:
            if not os.path.isfile(weights_path):
                errors_encountered += 1
                continue

            weights_matrix = np.loadtxt(weights_path, delimiter=",")

                        numeric_sample_data = pd.to_numeric(sample_data, errors='coerce').fillna(0)
            nonzero_species = numeric_sample_data[numeric_sample_data > 1e-9]
            species_list = nonzero_species.index.tolist()
            num_species = len(species_list)

            if weights_matrix.shape != (num_species, num_species):
                 errors_encountered +=1
                 continue

            G_filtered = nx.Graph()
            # Add nodes present in the original sample
            for species, abundance in nonzero_species.items():
                if pd.notna(species):
                    G_filtered.add_node(species, relab=abundance)

            # Add edges based on loaded weights and prevalence threshold
            edges_added = 0
            for r in range(num_species):
                for c in range(r + 1, num_species): 
                    weight = weights_matrix[r, c]
                    if np.isfinite(weight) and abs(weight) > edge_threshold:
                        node_r = species_list[r]
                        node_c = species_list[c]
                        if G_filtered.has_node(node_r) and G_filtered.has_node(node_c):
                            G_filtered.add_edge(node_r, node_c, weight=weight)
                            edges_added += 1

            # Save the filtered graph as GraphML only if it has edges
            if edges_added > 0:
                try:
                    nx.write_graphml(G_filtered, filtered_graph_path) 
                    graphs_saved += 1
                except Exception as save_err:
                     print(f"\n ERROR saving simple filtered graph {filtered_graph_path}: {save_err}")
                     errors_encountered += 1
  
            samples_processed += 1

        except Exception as e:
            errors_encountered += 1
            print(f"\n ERROR processing simple filtering for sample {sample_id}: {e}")
            # traceback.print_exc() # Uncomment for debugging

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Finished Simple Filtered Individual Graph Generation ---")
    print(f"  Total time: {duration:.2f} seconds")
    print(f"  Samples processed: {samples_processed}, Graphs saved: {graphs_saved}, Errors/Skipped: {errors_encountered}")


def aggregate_samples_for_condition(feature_table, metadata, condition_group, condition_id_col):
    """Aggregate samples by averaging for each condition."""
    try:
        samples_in_condition = metadata[metadata[condition_id_col] == condition_group].index
    except KeyError:
        print(f" Warning: Condition column '{condition_id_col}' not found during aggregation for {condition_group}.")
        return pd.Series(dtype=float), pd.Index([])
    except Exception as e:
         print(f" Error selecting samples for condition '{condition_group}' in aggregation: {e}")
         return pd.Series(dtype=float), pd.Index([])

    valid_samples = samples_in_condition.intersection(feature_table.index)
    if valid_samples.empty:
        print(f" No valid samples found for condition '{condition_group}' in aggregation.")
        return pd.Series(dtype=float), pd.Index([])

    condition_feature_table = feature_table.loc[valid_samples]
    if condition_feature_table.empty:
         print(f" Feature table subset is empty for condition '{condition_group}' in aggregation.")
         return pd.Series(dtype=float), valid_samples

    aggregated_feature_table = condition_feature_table.mean(axis=0).fillna(0)
    return aggregated_feature_table, valid_samples

#Create groupred graphs
def create_graph_from_aggregated_data(aggregated_data, threshold):
    """Create a graph based on aggregated data and edge weight threshold."""
    G = nx.Graph()
    numeric_agg_data = pd.to_numeric(aggregated_data, errors='coerce').fillna(0)
    nonzero_species = numeric_agg_data[numeric_agg_data > 1e-9]

    if nonzero_species.empty: return G

    species_list = nonzero_species.index.tolist()
    num_species = len(species_list)

    # Add nodes
    for species, abundance in nonzero_species.items():
        if pd.notna(species):
             G.add_node(species, relab=abundance) # Use 'relab' for mean abundance here
        else:
             print("Warning: Skipping node with invalid name (NaN?) in aggregated graph.")

    if num_species < 2: return G

    # Calculate weights using the same function but on mean abundances
    abundance_array = nonzero_species.to_numpy()
    try:
        weights_matrix, _ = compute_sample_weights(abundance_array)
        if weights_matrix.shape != (num_species, num_species):
             print(f" Warning: Weight matrix shape mismatch in aggregated graph. Expected ({num_species}, {num_species}), got {weights_matrix.shape}.")
             return G # Return graph with nodes only
    except Exception as e:
         print(f" Error calculating weights for aggregated data: {e}")
         return G # Return graph with nodes only

    # Add edges based on threshold
    for i in range(num_species):
        for j in range(i + 1, num_species):
            weight = weights_matrix[i, j]
            node_i = species_list[i]
            node_j = species_list[j]
            if G.has_node(node_i) and G.has_node(node_j):
                if np.isfinite(weight) and abs(weight) > threshold: # Apply prevalence threshold
                    G.add_edge(node_i, node_j, weight=weight)
    return G

def filter_feature_table(feature_table, graph_nodes):
    """Filter feature table columns to include only graph nodes."""
    valid_nodes = [node for node in graph_nodes if node in feature_table.columns]
    if not valid_nodes:
        return pd.DataFrame(index=feature_table.index)
    return feature_table[valid_nodes]

def create_simple_condition_graphs(feature_table, metadata, output_dir, condition_id_col, edge_threshold):
    """Creates and saves simple AGGREGATED graphs (GraphML) for each condition."""
    print(f"\n--- Generating Simple Aggregated Condition Graphs (GraphML) (Threshold > {edge_threshold}) ---")
    # Create specific directory using the threshold value
    base_condition_dir = os.path.join(output_dir, f"03_Simple_Filtering_(Threshold_{edge_threshold})", "Aggregated_Conditions")
    os.makedirs(base_condition_dir, exist_ok=True)
    print(f" Aggregated condition graphs will be saved in: '{base_condition_dir}'")

    try:
        conditions = metadata[condition_id_col].unique()
    except KeyError:
         print(f" ERROR: Condition ID column '{condition_id_col}' not found in metadata for aggregation.")
         return

    cond_iterator = tqdm_func(conditions, desc="Aggregated Conditions", leave=False)
    graphs_saved = 0
    tables_saved = 0

    for group in cond_iterator:
        safe_name = safe_filename(group)
        cond_out_dir = os.path.join(base_condition_dir, safe_name)
        os.makedirs(cond_out_dir, exist_ok=True)

        agg_data, samples = aggregate_samples_for_condition(feature_table, metadata, group, condition_id_col)

        if agg_data.empty or agg_data.sum() < 1e-9 :
            print(f" Skipping aggregated graph for condition '{group}': No aggregated data.")
            continue

        G_agg = create_graph_from_aggregated_data(agg_data, threshold=edge_threshold)

        if G_agg.number_of_nodes() == 0:
            print(f" Skipping saving aggregated graph for condition '{group}': No nodes in graph.")
            continue

        # Save graph files (
        try:
            # pkl_path = os.path.join(cond_out_dir, f"graph_agg.pkl") 
            gml_path = os.path.join(cond_out_dir, f"graph_agg.graphml")
            # with open(pkl_path, 'wb') as f: pickle.dump(G_agg, f)
            nx.write_graphml(G_agg, gml_path)
            graphs_saved += 1
        except Exception as e:
            print(f" Error saving aggregated graph for {group}: {e}")

        # Save filtered feature table for this condition based on aggregated graph nodes
        try:
            nodes = list(G_agg.nodes())
            valid_samples = samples.intersection(feature_table.index)
            if not valid_samples.empty and nodes:
                # Filter the original feature table, not an already filtered one
                ft_original_subset = feature_table.loc[valid_samples]
                ft_filt = filter_feature_table(ft_original_subset, nodes)
                if not ft_filt.empty:
                    ft_filt_path = os.path.join(cond_out_dir, f"feature_table_agg_filtered.csv")
                    ft_filt.to_csv(ft_filt_path)
                    tables_saved +=1
        except Exception as e:
            print(f" Error saving filtered table for aggregated {group}: {e}")

    print(f"--- Finished Simple AGGREGATED Condition Graph Generation ---")
    print(f"  GraphML Graphs saved: {graphs_saved}, Filtered Tables saved: {tables_saved}")


# Bootstrap P-Value Filtering Functions ---

def create_bootstrap_population(observed_data_df, condition_group, output_dir_base,
                                n_boots): 
    """Generates bootstrap weight matrices for a given condition's data."""
    print(f" Bootstrapping {n_boots} replicates for condition: {condition_group}")
    
    condition_results_dir = os.path.join(output_dir_base, "Intermediates", condition_group)
    matrices_dir = os.path.join(condition_results_dir, "matrices")
    os.makedirs(matrices_dir, exist_ok=True)
    print(f"   Bootstrap intermediate matrices will be saved in: {matrices_dir}")

    raw_data = observed_data_df.to_numpy()
    n_samples, num_species = raw_data.shape
    if n_samples == 0 or num_species == 0:
        print("   ERROR: No data (samples or species) provided for bootstrapping.")
        return None 

    # Generate bootstrap datasets (indices -> resampled data)
    bootstrap_indices_list = [np.random.choice(n_samples, size=n_samples, replace=True) for _ in range(n_boots)]
    bstrap_otus_datasets = [raw_data[indices] for indices in bootstrap_indices_list]

    # Function to process a single bootstrap replicate 
    def process_bootstrap_sample(b, otu_sample):
        try:
            if otu_sample.shape[1] == 0: 
                 return np.zeros((num_species, num_species))
            # Calculate average weights for the bootstrap sample
            w, _ = compute_all_weights(otu_sample)
            if w.shape == (num_species, num_species):
                 # Save individual matrix
                 np.savetxt(os.path.join(matrices_dir, f"bstrap_weight_matrix_{b}.csv"), w, delimiter=",")
                 return w
            else: 
                 return np.zeros((num_species, num_species))
        except Exception as e:
            print(f"   Error processing bootstrap replicate {b}: {e}")
            return np.zeros((num_species, num_species))

    # Run in parallel
    results_matrices = Parallel(n_jobs=-1)(delayed(process_bootstrap_sample)(b, otu)
                                            for b, otu in enumerate(tqdm_func(bstrap_otus_datasets, desc=f"  Bootstrapping {condition_group}", leave=False)))

    # Filter out potential None or incorrectly shaped arrays before stacking
    valid_results_matrices = [m for m in results_matrices if isinstance(m, np.ndarray) and m.shape == (num_species, num_species)]

    if not valid_results_matrices:
        print(f"   ERROR: No valid bootstrap matrices were generated for {condition_group}.")
        return None
    elif len(valid_results_matrices) < n_boots * 0.5: # Warning if less than half succeeded
        print(f"   WARNING: Only {len(valid_results_matrices)}/{n_boots} bootstrap replicates generated valid matrices for {condition_group}.")

    # Calculate element-wise mean and std deviation across valid bootstrap matrices
    print("   Calculating mean and std deviation across bootstrap matrices...")
    try:
        stacked_matrices = np.stack(valid_results_matrices, axis=0)
        bstrap_means = np.mean(stacked_matrices, axis=0)
        bstrap_stds = np.std(stacked_matrices, axis=0)

        # Save mean and std matrices
        np.savetxt(os.path.join(condition_results_dir, f"means.csv"), bstrap_means, delimiter=",")
        np.savetxt(os.path.join(condition_results_dir, f"stds.csv"), bstrap_stds, delimiter=",")
        print(f"   Saved mean and std matrices to: {condition_results_dir}")
        # Return the path to the results dir for the next step (contains matrices subdir, means, stds)
        return condition_results_dir
    except Exception as e:
        print(f"   ERROR calculating/saving mean/std matrices for {condition_group}: {e}")
        return None

def filtering_pvals_for_each_sample(df_cond, condition_group, bstrap_intermed_dir,
                                    output_dir_base, pval_thresh,
                                    delete_bootstrap_matrices=True):
    """
    Filters each sample's graph based on comparison to bootstrap distributions.
    Saves filtered weights matrix and graphml graph per sample.
    Saves condition-level summary and filtered OTU table.
    """
    print(f"  Filtering individual samples for condition: {condition_group} (p < {pval_thresh})")
    matrices_dir = os.path.join(bstrap_intermed_dir, "matrices") # Path to individual bootstrap matrices

    # Define output paths based on new structure
    condition_summary_dir = os.path.join(output_dir_base, "Condition_Summaries")
    sample_results_base_dir = os.path.join(output_dir_base, "Individual_Samples_by_Condition", condition_group)
    os.makedirs(condition_summary_dir, exist_ok=True)
    os.makedirs(sample_results_base_dir, exist_ok=True) # Dir for all samples of this condition

    # --- Load Bootstrap Results ---
    bs_data_3d = None
    num_replicates_loaded = 0
    expected_shape = None
    try:
        if not os.path.isdir(matrices_dir):
             print(f"   ERROR: Bootstrap matrices directory not found: {matrices_dir}")
             return None, None # Cannot proceed without bootstrap matrices
        bootstrap_files = [f for f in os.listdir(matrices_dir) if f.startswith('bstrap_weight_matrix_') and f.endswith('.csv')]
        if not bootstrap_files:
            print(f"   ERROR: No bootstrap matrix files found in {matrices_dir}")
            return None, None

        # Load matrices carefully into memory
        bootstrap_matrices = []
        for f in bootstrap_files:
            file_path = os.path.join(matrices_dir, f)
            try:
                m = np.loadtxt(file_path, delimiter=",")
                if m.ndim != 2: continue # Skip non-2D arrays
                if expected_shape is None: expected_shape = m.shape
                if m.shape == expected_shape: bootstrap_matrices.append(m)
            except Exception: pass # Ignore files that fail to load

        if not bootstrap_matrices:
             print(f"   ERROR: Could not load any valid bootstrap matrices from {matrices_dir}")
             return None, None
        bs_data_3d = np.stack(bootstrap_matrices, axis=0)
        num_replicates_loaded = bs_data_3d.shape[0]
        print(f"   Loaded {num_replicates_loaded} bootstrap matrices into memory.")

        # Delete individual bootstrap matrices if requestes
        if delete_bootstrap_matrices:
            print(f"    Attempting to delete individual bootstrap matrix files from {matrices_dir}...")
            deleted_count, error_count = 0, 0
            for f in bootstrap_files: 
                file_path = os.path.join(matrices_dir, f)
                try:
                    if os.path.exists(file_path): os.remove(file_path); deleted_count += 1
                except OSError: error_count += 1
            print(f"    Deletion attempt: {deleted_count} deleted, {error_count} errors.")
     
            try:
                if not os.listdir(matrices_dir): os.rmdir(matrices_dir)
            except OSError: pass 
        else:
             print(f"    Keeping individual bootstrap matrix files in {matrices_dir}.")

    except Exception as e:
        print(f"   ERROR during loading or initial processing of bootstrap matrices: {e}")
        return None, None


    data_cond = df_cond.to_numpy()
    num_samples, num_species = data_cond.shape

    # Verify dimension consistency
    if bs_data_3d is None or bs_data_3d.shape[1:] != (num_species, num_species): 
        print(f"   ERROR: Dimension mismatch or missing bootstrap data. Bootstrap shape {bs_data_3d.shape if bs_data_3d is not None else 'None'}, "
              f"condition data ({num_species} species). Cannot proceed.")
        return None, None

    species_names = df_cond.columns
    bs_weight_distributions = defaultdict(list)
    for i in range(num_species):
        for j in range(i + 1, num_species):
            weights_for_edge = bs_data_3d[:, i, j]
            bs_weight_distributions[i, j] = weights_for_edge[np.isfinite(weights_for_edge)].tolist()

    # Process samples
    print(f"   Processing {num_samples} samples...")
    filtering_summary_info = {}
    # Initialize filtered OTU table for the condition
    filtered_otu_table_data = df_cond.copy()
    samples_processed_count = 0

    sample_iterator = tqdm_func(range(num_samples), desc=f"    Filtering {condition_group}", leave=False, ncols=100)

    for counter in sample_iterator:
        sample_name = df_cond.index[counter]
        safe_sample_name = safe_filename(sample_name)
        sample_data_rel = data_cond[counter, :] # Relative abundances for the sample

        # Define sample-specific output directory
        sample_output_dir = os.path.join(sample_results_base_dir, safe_sample_name)
        os.makedirs(sample_output_dir, exist_ok=True)

        try:
            # Calculate unfiltered weights for THIS sample 
            sample_weights_unfiltered, _ = compute_sample_weights(sample_data_rel)
            if sample_weights_unfiltered.shape != (num_species, num_species):
                 print(f"    Warning: Unexpected unfiltered weight shape for sample {sample_name}. Skipping.")
                 continue # Skip to next sample

            # Filter edges based on bootstrap distributions
            filtered_sample_weights = sample_weights_unfiltered.copy() 
            total_edges_in_sample = 0
            edges_filtered_count = 0

            for i in range(num_species):
                for j in range(i + 1, num_species):
                    sample_w_ij = sample_weights_unfiltered[i, j]

                    if abs(sample_w_ij) > 1e-9 and np.isfinite(sample_w_ij): # Consider only present edges
                        total_edges_in_sample += 1
                        # Get bootstrap distribution for this edge
                        if (i, j) in bs_weight_distributions:
                            bs_dist = bs_weight_distributions[i, j]
                            # Perform t-test only if bootstrap distribution has sufficient data and variance
                            if len(bs_dist) > 1 and np.std(bs_dist) > 1e-9:
                                t_stat, p_val = stats.ttest_1samp(a=bs_dist, popmean=sample_w_ij, alternative='two-sided', nan_policy='omit')

                                # Filter if p < threshold (edge is significantly different from bootstrap mean)
                                if not np.isnan(p_val) and p_val < pval_thresh:
                                    filtered_sample_weights[i, j] = filtered_sample_weights[j, i] = 0
                                    edges_filtered_count += 1
                            

            # Save filtered weights matrix for the sample
            filtered_matrix_path = os.path.join(sample_output_dir, f"weights_filtered.csv")
            np.savetxt(filtered_matrix_path, filtered_sample_weights, delimiter=",")

            # --- Create and Save Filtered Graph for the sample ---
            G_filtered = nx.Graph()
            present_species_indices = np.where(sample_data_rel > 1e-9)[0]
            nodes_added = set()
            # Add nodes that were originally present in the sample
            for idx in present_species_indices:
                 if idx < len(species_names):
                      node_name = species_names[idx]
                      G_filtered.add_node(node_name, relab=sample_data_rel[idx])
                      nodes_added.add(node_name)

            # Add edges remaining after filtering
            edges_added_count = 0
            for i in range(num_species):
                for j in range(i + 1, num_species):
                    # Check weight is non-zero and finite
                    if abs(filtered_sample_weights[i, j]) > 1e-9 and np.isfinite(filtered_sample_weights[i, j]):
                         if i < len(species_names) and j < len(species_names):
                              node_i = species_names[i]; node_j = species_names[j]
                              if node_i in nodes_added and node_j in nodes_added:
                                   G_filtered.add_edge(node_i, node_j, weight=filtered_sample_weights[i, j])
                                   edges_added_count += 1

            # Save filtered graphml
            filtered_graph_path = os.path.join(sample_output_dir, f"graph_filtered.graphml") # Save as graphml
            try:
                # Remove isolated nodes before saving? Optional.
                # G_filtered.remove_nodes_from(isolated)
                nx.write_graphml(G_filtered, filtered_graph_path) # Use nx.write_graphml
            except Exception as graph_e:
                 print(f"    ERROR saving filtered graph for sample {sample_name}: {graph_e}")


            remaining_nodes = list(G_filtered.nodes())
            cols_to_zero_out = df_cond.columns[~df_cond.columns.isin(remaining_nodes)]
            if not cols_to_zero_out.empty:
                filtered_otu_table_data.loc[sample_name, cols_to_zero_out] = 0

            # Collect filtering summary
            filtering_summary_info[sample_name] = {
                "nodes_in": len(nodes_added), # Nodes initially present
                "edges_unfiltered": total_edges_in_sample,
                "edges_filtered_out": edges_filtered_count,
                "prop_filtered": (edges_filtered_count / total_edges_in_sample if total_edges_in_sample > 0 else 0),
                "nodes_out": G_filtered.number_of_nodes(), # Nodes remaining after filtering (potentially fewer if isolates removed)
                "edges_out": G_filtered.number_of_edges() # Edges remaining
            }
            samples_processed_count += 1

        except Exception as sample_proc_e:
            print(f"    ERROR processing sample {sample_name}: {sample_proc_e}")
            traceback.print_exc()

    print(f"   Saving condition-level summaries for {condition_group}...")
    summary_df = pd.DataFrame.from_dict(filtering_summary_info, orient="index")
    summary_csv_path = os.path.join(condition_summary_dir, f"filtering_summary_{condition_group}.csv")
    try:
        summary_df.to_csv(summary_csv_path)
        print(f"    Saved filtering summary: {summary_csv_path}")
    except Exception as summary_e:
        print(f"    ERROR saving filtering summary CSV: {summary_e}")

    filtered_otu_table_path = os.path.join(condition_summary_dir, f"feature_table_filtered_{condition_group}.csv")
    try:
        filtered_otu_table_data.to_csv(filtered_otu_table_path)
        print(f"    Filtered OTU table saved: {filtered_otu_table_path}")
    except Exception as otu_e:
        print(f"    ERROR saving filtered OTU table CSV: {otu_e}")

    print(f"  Finished filtering for condition {condition_group}. Samples processed: {samples_processed_count}/{num_samples}")
    return filtered_otu_table_data, summary_df


def run_bootstrap_filtering_per_condition(feature_table, metadata, output_dir,
                                          condition_id_col, num_bootstraps,
                                          p_value_threshold, min_samples_bootstrap):
    """Orchestrates bootstrapping and p-value filtering for each condition."""
    print(f"\n--- Running Bootstrap P-Value Filtering Workflow ---")
    
    # Create base directory for this run's parameters
    run_params_str = f"P_{p_value_threshold}_N_{num_bootstraps}"
    bootstrap_base_dir = os.path.join(output_dir, f"04_Bootstrap_Filtering_({run_params_str})")
    os.makedirs(bootstrap_base_dir, exist_ok=True) # Base for this specific run
    print(f" Bootstrap filtering outputs will be saved under: '{bootstrap_base_dir}'")

    try:
        all_conditions = metadata[condition_id_col].unique()
    except KeyError:
        print(f" ERROR: Condition ID column '{condition_id_col}' not found in metadata.")
        return {}, {} 

    print(f" Processing {len(all_conditions)} conditions based on: '{condition_id_col}'")
    print(f" Parameters: Bootstraps={num_bootstraps}, P-Value Threshold={p_value_threshold}, Min Samples={min_samples_bootstrap}")

    all_filtered_otus = {}
    all_summaries = {}
    conditions_processed_count = 0
    conditions_skipped_count = 0

    condition_iterator = tqdm_func(all_conditions, desc="Processing Conditions (Bootstrap Filter)", leave=True)

    for condition_group_raw in condition_iterator:
        condition_group = safe_filename(condition_group_raw) 
        print(f"\n {'='*10} Processing Condition: {condition_group_raw} (Safe name: {condition_group}) {'='*10}")

        # Select samples for this condition
        try:
            condition_sample_ids = metadata[metadata[condition_id_col] == condition_group_raw].index
        except Exception as e:
            print(f" Error selecting samples for condition '{condition_group_raw}': {e}. Skipping.")
            conditions_skipped_count += 1
            continue

        valid_condition_sample_ids = condition_sample_ids.intersection(feature_table.index)

        # Check sample size
        if len(valid_condition_sample_ids) < min_samples_bootstrap:
            print(f" Skipping condition '{condition_group_raw}': Only {len(valid_condition_sample_ids)} valid samples found (min required: {min_samples_bootstrap}).")
            conditions_skipped_count += 1
            continue

        df_cond = feature_table.loc[valid_condition_sample_ids]
        if df_cond.empty or df_cond.shape[1] == 0:
            print(f" Skipping condition '{condition_group_raw}': No valid data (samples or features) after selection.")
            conditions_skipped_count += 1
            continue
        print(f" Condition '{condition_group_raw}' has {len(df_cond)} samples and {df_cond.shape[1]} features for analysis.")

        # Run bootstrapping
        print("  Starting bootstrap population generation...")
        bstrap_intermed_dir = create_bootstrap_population(
            observed_data_df=df_cond,
            condition_group=condition_group, 
            output_dir_base=bootstrap_base_dir, # Pass the run-specific base dir
            n_boots=num_bootstraps,
            pval_thresh=p_value_threshold 
        )

        if bstrap_intermed_dir is None:
             print(f"  ERROR: Bootstrapping failed for condition '{condition_group}'. Skipping filtering step.")
             conditions_skipped_count += 1
             continue
        print("  Bootstrap population generation finished.")

        # Run filtering
        print("  Starting p-value filtering for samples...")
        filtered_otu_table, summary_df = filtering_pvals_for_each_sample(
            df_cond=df_cond,
            condition_group=condition_group, 
            bstrap_intermed_dir=bstrap_intermed_dir, 
            output_dir_base=bootstrap_base_dir, 
            pval_thresh=p_value_threshold
        )
        print("  P-value filtering for samples finished.")

        # Store results if successful
        if filtered_otu_table is not None:
             all_filtered_otus[condition_group_raw] = filtered_otu_table # Use original name as key
        if summary_df is not None:
             all_summaries[condition_group_raw] = summary_df 
        conditions_processed_count += 1

    print("\n" + "="*50)
    print("Bootstrap P-Value Filtering Workflow Finished.")
    print(f" Conditions processed: {conditions_processed_count}")
    print(f" Conditions skipped (due to sample size or errors): {conditions_skipped_count}")
    print("="*50)
    return all_filtered_otus, all_summaries


# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    print("--- Configuration Settings ---")

    otu_file_path = '/Users/nandini.gadhia/Documents/projects/gp_omics/data/rvc/OTU_table_full.csv'
    metadata_file_path = '/Users/nandini.gadhia/Documents/projects/gp_omics/data/rvc/metadata.tsv'
    sample_id_col = 'Sample-ID'
    condition_id_col = 'Group ID'

    # Base output directory for all results
    output_dir = "output/microbiome_network_analysis_outputs"

    # Which steps to run?
    run_feature_processing = True # Always needed
    run_unfiltered_analysis = True # Generates unfiltered graphs/weights
    run_simple_filtering = True # Prevalence filtering
    run_bootstrap_pval_filtering = True # Bootstrap p-value filtering

  
    simple_graph_edge_threshold = 0.1

    # Bootstrap P-Value Filtering parameters
    num_bootstraps_pval = 1000
    pval_threshold = 0.1 
    min_samples_pval = 5

    # Print Configuration 
    print(f"OTU Table Path: {otu_file_path}")
    print(f"Metadata Path: {metadata_file_path}")
    print(f"Sample ID Column: '{sample_id_col}'")
    print(f"Condition ID Column: '{condition_id_col}'")
    print(f"Base Output Directory: '{output_dir}'")
    print(f"\nWorkflow Selection:")
    print(f"  Run Feature Processing & Alignment: {run_feature_processing}")
    print(f"  Generate Unfiltered Results per Sample: {run_unfiltered_analysis}")
    print(f"  Run Simple Filtering (Threshold={simple_graph_edge_threshold}): {run_simple_filtering}")
    print(f"  Run Bootstrap P-Value Filtering (p<{pval_threshold}, N={num_bootstraps_pval}): {run_bootstrap_pval_filtering}")
    print("-" * 30)

    # Start Workflow 
    print("\n--- Starting Data Processing and Analysis Workflow ---")
    print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    overall_start_time = time.time()


    feature_table = None
    metadata = None

    try:
        # Create base output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory confirmed: '{os.path.abspath(output_dir)}'")

        # 1. Process Data (and save results)
        if run_feature_processing:
            print("\n[Step 1] Processing input data...")
            feature_table, metadata = process_data(
                otu_file_path, metadata_file_path, output_dir,
                sample_id=sample_id_col,
                condition_id=condition_id_col
            )
            # Basic check if processing was successful
            if feature_table is None or metadata is None or feature_table.empty or metadata.empty:
                 raise ValueError("Data processing failed to return valid feature table or metadata.")
            print("[Step 1] Data processing finished.")
            print("-"*50)
        else:
            print("\n[Step 1] Processing input data... SKIPPED")
            # Attempt to load previously processed data if skipping
            try:
                print("   Attempting to load previously processed data...")
                ft_path = os.path.join(output_dir, "01_Processed_Data", "feature_table_normalized.csv")
                meta_path = os.path.join(output_dir, "01_Processed_Data", "metadata_aligned.csv")
                feature_table = pd.read_csv(ft_path, index_col=0)
                metadata = pd.read_csv(meta_path, index_col=0)
              
                feature_table.index = feature_table.index.astype(str)
                metadata.index = metadata.index.astype(str)
                print(f"   Loaded feature table {feature_table.shape} and metadata {metadata.shape}")
            except FileNotFoundError:
                print("   ERROR: Could not load previously processed data. Please run Step 1 first.")
                raise 
            print("-"*50)


        # 2. Create and Save Unfiltered Sample Results (Graphs & Weights)
        if run_unfiltered_analysis:
            print("\n[Step 2] Creating Unfiltered Sample Results...")
            if feature_table is None: raise ValueError("Feature table not loaded for Step 2.")
            create_and_save_unfiltered_sample_results(
                feature_table,
                output_dir
            )
            print("[Step 2] Unfiltered sample results generation finished.")
            print("-"*50)
        else:
             print("\n[Step 2] Creating Unfiltered Sample Results... SKIPPED")
             print("-"*50)


        # 3. Run prevalence Filtering
        if run_simple_filtering:
            print(f"\n[Step 3] Applying Simple Filtering (Threshold = {simple_graph_edge_threshold})...")
            if feature_table is None: raise ValueError("Feature table not loaded for Step 3.")
            if metadata is None: raise ValueError("Metadata not loaded for Step 3.")

            save_simple_filtered_individual_graphs(
                feature_table,
                output_dir,
                edge_threshold=simple_graph_edge_threshold
            )
            
            create_simple_condition_graphs(
                feature_table,
                metadata,
                output_dir,
                condition_id_col=condition_id_col,
                edge_threshold=simple_graph_edge_threshold
            )
            print("[Step 3] Simple filtering finished.")
            print("-"*50)
        else:
             print("\n[Step 3] Applying Simple Filtering... SKIPPED")
             print("-"*50)


        # 4. Run Bootstrap P-Value Filtering
        if run_bootstrap_pval_filtering:
            print(f"\n[Step 4] Running Bootstrap P-Value Filtering (p < {pval_threshold}, N = {num_bootstraps_pval})...")
            if feature_table is None: raise ValueError("Feature table not loaded for Step 4.")
            if metadata is None: raise ValueError("Metadata not loaded for Step 4.")

            filtered_otu_tables_by_cond, filter_summaries_by_cond = run_bootstrap_filtering_per_condition(
                feature_table=feature_table,
                metadata=metadata,
                output_dir=output_dir,
                condition_id_col=condition_id_col,
                num_bootstraps=num_bootstraps_pval,
                p_value_threshold=pval_threshold,
                min_samples_bootstrap=min_samples_pval
            )
            print("[Step 4] Bootstrap P-Value Filtering finished.")
            
            # Example to access results if needed later
            # if filter_summaries_by_cond:
            #     print("   Generated filter summaries for conditions:", list(filter_summaries_by_cond.keys()))
            print("-"*50)
        else:
             print("\n[Step 4] Bootstrap P-Value Filtering Workflow... SKIPPED")
             print("-"*50)


        # End Workflow 
        overall_end_time = time.time()
        print("\n" + "="*50)
        print("--- Workflow Finished ---")
        print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")
        print(f"Output saved in base directory: {os.path.abspath(output_dir)}")
        print("Check subdirectories for results from enabled analyses:")
        
        if run_feature_processing: print(" - 01_Processed_Data/")
        if run_unfiltered_analysis: print(" - 02_Unfiltered_Sample_Results/")
        if run_simple_filtering: print(f" - 03_Simple_Filtering_(Threshold_{simple_graph_edge_threshold})/")
        if run_bootstrap_pval_filtering: print(f" - 04_Bootstrap_Filtering_(P_{pval_threshold}_N_{num_bootstraps_pval})/")
        print("="*50)

    # Errors
    except FileNotFoundError as e:
        print(f"\n--- FATAL ERROR: Input file not found ---", flush=True)
        print(e); print("Please check paths in the Configuration section.", flush=True)
    except ValueError as ve:
        print(f"\n--- FATAL ERROR: Data validation or processing error ---", flush=True)
        print(ve); print("Check input file formats, column names, or script logic.", flush=True)
    except KeyError as ke:
        print(f"\n--- FATAL ERROR: Column/Key not found ---", flush=True)
        print(f"Column or Key '{ke}' not found. Check config or data integrity.", flush=True)
    except MemoryError:
         print(f"\n--- FATAL ERROR: Insufficient memory ---", flush=True)
         print("Consider using a machine with more RAM, especially for bootstrapping.", flush=True)
    except Exception as e:
        print(f"\n--- An unexpected FATAL error occurred ---", flush=True)
        traceback.print_exc()

    finally:
        print(f"\nScript finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")