
import sys
import os
import argparse
import shutil

# Add project root to path to fix module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import fiftyone as fo
    import fiftyone.brain as fob
except ImportError:
    print("Error: fiftyone not found. Please install it with `pip install fiftyone`.")
    sys.exit(1)

# PATCH: Define dummy RunResults class to allow unpickling of old datasets
# The user encountered "module 'visualization.Visualization' has no attribute 'RunResults'"
try:
    import visualization.voxel51_vis
    class RunResults:
        pass
    if not hasattr(visualization.voxel51_vis, 'RunResults'):
        setattr(visualization.voxel51_vis, 'RunResults', RunResults)
except ImportError:
    pass # If we can't import it, we can't patch it.


def get_all_snapshots(search_dirs=["./exp"]):
    found_snapshots = []
    
    # Check if ./exp exists, else use .
    if not os.path.exists("./exp") and "./exp" in search_dirs:
         search_dirs = [d for d in search_dirs if d != "./exp"]
         if not search_dirs:
             search_dirs = ["."]

    for root_search in search_dirs:
        if not os.path.exists(root_search):
            continue
        for root, dirs, files in os.walk(root_search):
            if "visualization_snapshots" in dirs:
                snap_dir = os.path.join(root, "visualization_snapshots")
                # Check for exported FO datasets
                for f in os.listdir(snap_dir):
                    if f.endswith("_snapshot.pkl"):
                        # The corresponding FO export should be there too
                        dataset_name = f.replace("_snapshot.pkl", "")
                        fo_export_dir = os.path.join(snap_dir, f"{dataset_name}_fo_export")
                        if os.path.isdir(fo_export_dir):
                            found_snapshots.append({
                                "name": dataset_name,
                                "pkl_path": os.path.join(snap_dir, f),
                                "fo_path": fo_export_dir,
                                "timestamp": os.path.getmtime(os.path.join(snap_dir, f))
                            })
                            
    # Sort by time (newest first)
    found_snapshots.sort(key=lambda x: x["timestamp"], reverse=True)
    return found_snapshots


def delete_associated_run_data(dataset_name, all_snapshots, force, auto_delete):
    snapshot = next((s for s in all_snapshots if s["name"] == dataset_name), None)
    if not snapshot:
        return
    
    fo_export_dir = snapshot["fo_path"]
    viz_dir = os.path.dirname(fo_export_dir)
    run_dir = os.path.dirname(viz_dir)
    
    if os.path.basename(viz_dir) != "visualization_snapshots":
        return
        
    if not os.path.exists(run_dir):
        return
        
    if auto_delete:
        print(f"Deleting associated run directory: {run_dir}")
        shutil.rmtree(run_dir)
        return
        
    if not force:
        confirm = input(f"Do you also want to delete the associated experiment run directory on disk: {run_dir}? (y/N): ")
        if confirm.lower() == 'y':
            try:
                shutil.rmtree(run_dir)
                print(f"Deleted run directory: {run_dir}")
            except Exception as e:
                print(f"Failed to delete {run_dir}: {e}")
    else:
        # If force was true but auto_delete wasn't explicitly requested, we skip.
        pass

def main():
    parser = argparse.ArgumentParser(
        description="Cleanup FiftyOne datasets and optionally view one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python visualization/cleanup.py                    # List all datasets
  uv run python visualization/cleanup.py my_dataset         # View 'my_dataset'
  uv run python visualization/cleanup.py --delete-all       # Delete ALL datasets (careful!)
  uv run python visualization/cleanup.py my_dataset --delete-others  # View 'my_dataset', delete others
        """
    )
    parser.add_argument("view", nargs="?", help="Name of the dataset to view (and keep).")
    parser.add_argument("--delete", type=str, help="Delete a specific dataset by name.")
    parser.add_argument("--delete-others", action="store_true", help="Delete all other datasets except the one specified in 'view'.")
    parser.add_argument("--delete-all", action="store_true", help="Delete ALL datasets. Cannot be used with 'view'.")
    parser.add_argument("--delete-run-data", action="store_true", help="Automatically delete associated experiment physical folders from ./exp/ when deleting fiftyone datasets.")
    parser.add_argument("--force", action="store_true", help="Force deletion without confirmation prompt.")
    parser.add_argument("--list-snapshots", action="store_true", help="List available visualization snapshots in ./exp/ directories.")
    parser.add_argument("--load-snapshot", type=str, help="Load a snapshot by path or name (partial match).")
    
    # Advanced features
    parser.add_argument("--compute-uniqueness", action="store_true", help="Compute uniqueness scores for the dataset.")
    parser.add_argument("--compute-mistakenness", action="store_true", help="Compute mistakenness for all prediction fields.")
    parser.add_argument("--compute-metadata", action="store_true", help="Compute image metadata (width, height, channels, etc.).")

    args = parser.parse_args()

    # --delete validation
    if (args.delete_all or args.delete) and (args.view or args.load_snapshot):
        print("Error: Cannot use --delete-all or --delete with view or load operations.")
        sys.exit(1)

    all_datasets = fo.list_datasets()
    
    # ----------------------------------------------------------------------
    # List Snapshots Logic
    # ----------------------------------------------------------------------
    if args.list_snapshots:
        print("Searching for snapshots in ./exp/ ...")
        found_snapshots = get_all_snapshots()
        
        if not found_snapshots:
            print("No snapshots found.")
        else:
            print(f"Found {len(found_snapshots)} snapshots:")
            for i, snap in enumerate(found_snapshots):
                print(f"  [{i}] {snap['name']}")
                print(f"      Path: {snap['pkl_path']}")
        return

    # ----------------------------------------------------------------------
    # Load Snapshot Logic
    # ----------------------------------------------------------------------
    if args.load_snapshot:
        target = args.load_snapshot
        
        # 1. Resolve target to a path
        fo_export_dir = None
        
        # If it's a direct path
        if os.path.isdir(target) and target.endswith("_fo_export"):
            fo_export_dir = target
        elif os.path.isfile(target) and target.endswith(".pkl"):
             # User pointed to pickle, infer export dir
             fo_export_dir = target.replace("_snapshot.pkl", "_fo_export")
        else:
            # Search for it
            print(f"Searching for snapshot matching '{target}' in ./exp/ ...")
            # Deduplicate search
            search_dirs = ["./exp"]
            if not os.path.exists("./exp"):
                 search_dirs = ["."]
            
            matches = []
            seen_matches = set()

            for root_search in search_dirs:
                if not os.path.exists(root_search): continue
                for root, dirs, files in os.walk(root_search):
                    if "visualization_snapshots" in dirs:
                        snap_dir = os.path.join(root, "visualization_snapshots")
                        for d in os.listdir(snap_dir):
                             # Check directories provided they look like exports
                             if os.path.isdir(os.path.join(snap_dir, d)) and d.endswith("_fo_export"):
                                 # Check if name matches
                                 if target in d:
                                     full_path = os.path.abspath(os.path.join(snap_dir, d))
                                     if full_path not in seen_matches:
                                         matches.append(full_path)
                                         seen_matches.add(full_path)
            
            if not matches:
                print(f"Error: No snapshot found matching '{target}'")
                sys.exit(1)
            elif len(matches) > 1:
                print(f"Error: Multiple snapshots match '{target}':")
                for m in matches: print(f"  - {m}")
                print("Please be more specific.")
                sys.exit(1)
            else:
                fo_export_dir = matches[0]

        if not os.path.isdir(fo_export_dir):
            print(f"Error: Export directory not found: {fo_export_dir}")
            sys.exit(1)

        print(f"Loading snapshot from: {fo_export_dir}")
        dataset_name = os.path.basename(fo_export_dir).replace("_fo_export", "")
        
        # Check if dataset already exists
        if dataset_name in all_datasets:
            print(f"Dataset '{dataset_name}' already exists in FiftyOne.")
            if not args.force:
                choice = input("Overwrite? (y/N): ")
                if choice.lower() != 'y':
                    print("Using existing dataset.")
                    # Fallthrough to launch
                    args.view = dataset_name # Set view arg for launch block below
                else:
                    fo.delete_dataset(dataset_name)
                    print(f"Deleted existing '{dataset_name}'. Importing new one...")
                    ds = fo.Dataset.from_dir(
                        dataset_dir=fo_export_dir,
                        dataset_type=fo.types.FiftyOneDataset,
                        name=dataset_name # Force name
                    )
                    args.view = dataset_name
            else:
                # Force overwrite
                fo.delete_dataset(dataset_name)
                ds = fo.Dataset.from_dir(
                    dataset_dir=fo_export_dir,
                    dataset_type=fo.types.FiftyOneDataset,
                    name=dataset_name
                )
                args.view = dataset_name
        else:
            # Import
            print("Importing dataset...")
            try:
                ds = fo.Dataset.from_dir(
                    dataset_dir=fo_export_dir,
                    dataset_type=fo.types.FiftyOneDataset,
                    name=dataset_name
                )
                print(f"Successfully imported '{dataset_name}'.")
                args.view = dataset_name
            except Exception as e:
                print(f"Import failed: {e}")
                sys.exit(1)

    # ----------------------------------------------------------------------
    # Listing Logic (Default)
    # ----------------------------------------------------------------------
    # If no arguments (and we didn't just load something), just list
    if not args.view and not args.delete_all and not args.delete and not args.list_snapshots and not args.load_snapshot:
        print("Existing datasets:")
        if not all_datasets:
            print("  (No datasets found)")
        for name in all_datasets:
            print(f"  - {name}")
        print("\nSee --help for usage.")
        return

    # Delete All Logic
    if args.delete_all:
        if not all_datasets:
            print("No datasets to delete.")
            return
            
        print(f"\nWARNING: The following {len(all_datasets)} datasets will be DELETED:")
        print(all_datasets)
        
        if not args.force:
            confirm = input("Are you sure? This cannot be undone. (y/N): ")
            if confirm.lower() != 'y':
                print("Aborting.")
                sys.exit(0)
        
        # Cache snapshots info for cleanup
        snaps = get_all_snapshots()
        
        for name in all_datasets:
            try:
                fo.delete_dataset(name)
                print(f"Deleted dataset {name}")
                delete_associated_run_data(name, snaps, args.force, args.delete_run_data)
            except Exception as e:
                print(f"Failed to delete {name}: {e}")
        return

    # Delete single Logic
    if args.delete:
        if args.delete not in all_datasets:
            print(f"Dataset '{args.delete}' not found in FiftyOne.")
            # Even if it's not in fiftyone, try to find and delete its disk representation? 
            # We'll just list it from snapshots to be safe.
            snaps = get_all_snapshots()
            delete_associated_run_data(args.delete, snaps, args.force, args.delete_run_data)
            return

        if not args.force:
            confirm = input(f"Are you sure you want to delete dataset '{args.delete}'? (y/N): ")
            if confirm.lower() != 'y':
                print("Aborting.")
                sys.exit(0)
                
        # Cache snapshots info
        snaps = get_all_snapshots()
        try:
            fo.delete_dataset(args.delete)
            print(f"Deleted dataset {args.delete}")
            delete_associated_run_data(args.delete, snaps, args.force, args.delete_run_data)
        except Exception as e:
            print(f"Failed to delete {args.delete}: {e}")
        return

    # View Logic (with optional delete-others)
    if args.view:
        if args.view not in fo.list_datasets(): # Check again in case we just imported
            print(f"Error: Dataset '{args.view}' not found.")
            print("Available datasets:", fo.list_datasets())
            sys.exit(1)

        if args.delete_others:
            current_all = fo.list_datasets()
            to_delete = [d for d in current_all if d != args.view]
            if to_delete:
                print(f"\nThe following {len(to_delete)} datasets will be DELETED:")
                print(to_delete)
                if not args.force:
                    confirm = input(f"Are you sure you want to keep '{args.view}' and delete the rest? (y/N): ")
                    if confirm.lower() != 'y':
                        print("Aborting logic. No datasets were deleted.")
                        sys.exit(0)
                
                snaps = get_all_snapshots()
                for name in to_delete:
                    try:
                        fo.delete_dataset(name)
                        print(f"Deleted dataset {name}")
                        delete_associated_run_data(name, snaps, args.force, args.delete_run_data)
                    except Exception as e:
                        print(f"Failed to delete {name}: {e}")
            else:
                print("\nNo other datasets to delete.")

        print(f"\nLaunching app for '{args.view}'... (Ctrl+C to stop)")
        try:
            ds = fo.load_dataset(args.view)
            
            # ------------------------------------------------------------------
            # Advanced Analysis Logic
            # ------------------------------------------------------------------
            
            # 1. Metadata
            if args.compute_metadata:
                print("Computing image metadata...")
                ds.compute_metadata()
                print("Metadata computed.")

            # 2. Uniqueness
            if args.compute_uniqueness:
                print("Computing uniqueness...")
                # Try to use existing embeddings if available
                brain_key = "uniqueness"
                if brain_key in ds.list_brain_runs():
                    print(f"Uniqueness run '{brain_key}' already exists.")
                else:
                    # Check for embeddings field
                    # visualization/Visualization.py saves HF embeddings as "hf_ground_truth"
                    # But wait, add_run stores them in memory, are they on the samples?
                    # The samples usually don't have the embeddings directly unless we added them.
                    # Visualization.py keeps them in memory or creates brain runs.
                    # Brain runs store embeddings in the brain_results, but we can also use 'compute_visualization' embeddings.
                    # Let's see if we can find any embeddings.
                    
                    found_embeddings = False
                    # Check if 'hf_ground_truth' brain key exists (from UMAP/t-SNE)
                    # Actually, fob.compute_visualization stores the visualization, not necessarily the raw embeddings 
                    # as a field relative to the sample unless we requested it.
                    # However, we can re-use the *model* if we had it, but we are offline here.
                    # Ideally, uniqueness needs a model or pre-computed embeddings.
                    # If we don't have embeddings, we can't easily compute uniqueness without loading a model.
                    # fob.compute_uniqueness can use a default model (mobilenet-v2) if no embeddings provided.
                    # This is heavy but works.
                    
                    print("Computing uniqueness (this may download a model if no embeddings found)...")
                    try:
                         fob.compute_uniqueness(ds, uniqueness_field="uniqueness")
                         print("Uniqueness computed.")
                    except Exception as e:
                         print(f"Failed to compute uniqueness: {e}")

            # 3. Mistakenness
            if args.compute_mistakenness:
                print("Computing mistakenness...")
                # Find all Classification fields
                pred_fields = []
                for field in ds.get_field_schema():
                    if isinstance(ds.get_field_schema()[field], fo.EmbeddedDocumentField):
                         # predictions are usually EmbeddedDocumentField(fo.Classification) or fo.Classifications
                         pass
                
                # In Visualization.py, predictions are added as labels. 
                # e.g. run["selection_method"] which is the label field name?
                # Let's inspect potential label fields.
                # Common pattern in this repo: "selected_{method}_E{epoch}"? 
                # Or just the method name? 
                # Visualization.py: visualizer.add_run(..., labels=preds, ...)
                # process_milestones: labels=preds. 
                # It seems they are added to the dataset?
                # Actually, `add_run` stores them in `self.runs`. 
                # And `Visualization.py` later calls `_export_snapshot`. 
                # BUT, does it add them to the FiftyOne dataset?
                # `process_milestones` calls `visualizer.add_run`.
                # `compute_all_visualizations` computes UMAP.
                # `run_final_evaluations` creates `fo.Classifications` and saves them to the dataset!
                # It uses `fkey = f"{run['selection_method']}_E{run['epoch']}"`.
                
                # So we look for fields that look like prediction fields
                candidates = [f for f in ds.get_field_schema() if "_E" in f]
                
                if not candidates:
                    print("No obvious prediction fields found (looking for '*_E*').")
                else:
                    print(f"Found {len(candidates)} prediction fields: {candidates}")
                    # We need ground truth. Is it 'ground_truth' or 'target'?
                    # Visualization.py uses "ground_truth" or "target"?
                    # Inspecting dataset to find GT.
                    gt_field = "ground_truth"
                    if gt_field not in ds.get_field_schema():
                        if "target" in ds.get_field_schema():
                            gt_field = "target"
                        else:
                            print("Could not identify ground truth field (expected 'ground_truth' or 'target'). Skipping mistakenness.")
                            gt_field = None
                    
                    if gt_field:
                        for pred_field in candidates:
                            print(f"Computing mistakenness for {pred_field} vs {gt_field}...")
                            try:
                                fob.compute_mistakenness(
                                    ds, 
                                    pred_field=pred_field, 
                                    label_field=gt_field,
                                    mistakenness_field=f"mistakenness_{pred_field}"
                                )
                            except Exception as e:
                                print(f"  Failed: {e}")
                        print("Mistakenness computed.")

            # ------------------------------------------------------------------
            
            session = fo.launch_app(ds)
            print("App launched. Open http://localhost:5151")
            session.wait()
        except Exception as e:
             # Ensure we catch weird FO errors like the import one before
             print(f"Error launching app or loading dataset: {e}")
             import traceback
             traceback.print_exc()

if __name__ == "__main__":
    main()
