# src/validate_data.py
import os
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
import glob
from PIL import Image
import numpy as np
import pandas as pd

# Context for GX (in-memory)
# gx.get_context will initialize a basic project structure in ./great_expectations if it doesn't exist
context = gx.get_context(project_root_dir='.') 

# Define paths
TRAIN_DATA_DIR_GX = "data_train_engine/train" # Points to the parent of NORMAL/PNEUMONIA
EXPECTATION_SUITE_NAME = "image_data_validation_suite"

# Helper to load and preprocess a sample of images
def load_sample_images_for_validation(data_dir, num_samples_per_class=5, target_size=(150,150)):
    image_data_list = []
    if not os.path.exists(data_dir):
        print(f"Error: Data directory for validation not found: {data_dir}")
        return pd.DataFrame(image_data_list)
        
    for class_name in os.listdir(data_dir): # e.g., NORMAL, PNEUMONIA
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            image_files = glob.glob(os.path.join(class_path, "*.jpg")) + \
                          glob.glob(os.path.join(class_path, "*.jpeg")) + \
                          glob.glob(os.path.join(class_path, "*.png"))
            
            for i, img_file in enumerate(image_files):
                if i >= num_samples_per_class:
                    break
                try:
                    with Image.open(img_file) as img:
                        # Basic info
                        width, height = img.size
                        mode = img.mode
                        
                        # Simulate preprocessing (resize and normalize)
                        img_rgb = img
                        if img.mode == 'RGBA':
                            img_rgb = img.convert('RGB')
                        
                        img_resized = img_rgb.resize(target_size)
                        img_array = np.array(img_resized)
                        
                        pixel_mean = img_array.mean()
                        pixel_min = img_array.min()
                        pixel_max = img_array.max()
                        num_channels = img_array.shape[2] if len(img_array.shape) == 3 else 1

                        image_data_list.append({
                            "filename": os.path.basename(img_file),
                            "width": width, # Original width
                            "height": height, # Original height
                            "mode": mode,
                            "resized_width": img_resized.width,
                            "resized_height": img_resized.height,
                            "num_channels_resized": num_channels,
                            "pixel_mean_resized": pixel_mean,
                            "pixel_min_resized": pixel_min,
                            "pixel_max_resized": pixel_max,
                            "class": class_name
                        })
                except Exception as e:
                    print(f"Could not process {img_file}: {e}")
    return pd.DataFrame(image_data_list)

def create_or_get_expectation_suite(suite_name, target_width=150, target_height=150):
    try:
        # Attempt to get the suite. If it exists, we might overwrite it or just load it.
        # For this script, we'll define it each time to ensure it has the latest expectations.
        # context.delete_expectation_suite(expectation_suite_name=suite_name) # Optional: uncomment to always recreate
        suite = context.add_expectation_suite(expectation_suite_name=suite_name, overwrite_existing=True)
        print(f"Created/Replaced expectation suite: {suite_name}")
    except Exception as e_get_suite: # More general exception if specific one is unknown
        print(f"Error getting/creating suite, fallback to add_or_update: {e_get_suite}")
        # Fallback or ensure creation if get_expectation_suite fails to find non-existent and add_expectation_suite is preferred
        suite = context.add_or_update_expectation_suite(expectation_suite_name=suite_name)
        print(f"Created/Updated expectation suite via add_or_update: {suite_name}")


    # Basic Expectations for image properties (after resizing)
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "resized_width", "value_set": [target_width]}
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "resized_height", "value_set": [target_height]}
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "num_channels_resized", "value_set": [3]} # Assuming RGB
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "pixel_min_resized", 
                "min_value": 0, "max_value": 255, # Before normalization
                "mostly": 0.95 # Allow some outliers if needed, but for raw images, this should be strict.
            }
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "pixel_max_resized", 
                "min_value": 0, "max_value": 255, # Before normalization
                "mostly": 0.95
            }
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_mean_to_be_between",
            kwargs={
                "column": "pixel_mean_resized", 
                "min_value": 1, # Avoid totally black images (min_value >= 0 usually)
                "max_value": 254 # Avoid totally white images (max_value <= 255 usually)
            }
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "filename"}
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "class", "value_set": ["NORMAL", "PNEUMONIA"]} # Based on dummy data
        )
    )
    # Save the suite
    context.save_expectation_suite(expectation_suite=suite, expectation_suite_name=suite_name)
    return suite

def validate_data_sample(suite_name, image_df):
    if image_df.empty:
        print("No image data to validate.")
        return None

    batch_request = RuntimeBatchRequest(
        datasource_name="runtime_datasource_pandas", # Updated for clarity
        data_connector_name="runtime_data_connector_pandas", # Updated for clarity
        data_asset_name="image_sample_data_asset", # Updated for clarity
        runtime_parameters={"batch_data": image_df}, 
        batch_identifiers={"id": "default_runtime_identifier"} 
    )
    
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )
    validation_result = validator.validate()
    return validation_result

if __name__ == "__main__":
    # Target size used in train_engine.py
    TARGET_IMG_WIDTH = 150
    TARGET_IMG_HEIGHT = 150

    # 1. Ensure dummy data exists for this script to run
    #    train_engine.py should create data_train_engine/train/NORMAL and /PNEUMONIA
    normal_class_dir = os.path.join(TRAIN_DATA_DIR_GX, 'NORMAL')
    pneumonia_class_dir = os.path.join(TRAIN_DATA_DIR_GX, 'PNEUMONIA')

    if not (os.path.exists(normal_class_dir) and any(os.scandir(normal_class_dir)) and \
            os.path.exists(pneumonia_class_dir) and any(os.scandir(pneumonia_class_dir))):
        print(f"Dummy data not found or incomplete in {TRAIN_DATA_DIR_GX}.")
        print("Please run src/train_engine.py first to generate it (it creates dummy data).")
        print("Skipping Great Expectations validation demo.")
    else:
        print(f"Loading sample images from {TRAIN_DATA_DIR_GX} for validation demo...")
        sample_df = load_sample_images_for_validation(
            TRAIN_DATA_DIR_GX, 
            num_samples_per_class=5, 
            target_size=(TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT)
        )
        
        if not sample_df.empty:
            print(f"Loaded {len(sample_df)} images into DataFrame for validation.")
            print("Sample DataFrame head:")
            print(sample_df.head())
            
            print(f"\nCreating/Loading expectation suite: {EXPECTATION_SUITE_NAME}")
            expectation_suite = create_or_get_expectation_suite(
                EXPECTATION_SUITE_NAME,
                target_width=TARGET_IMG_WIDTH,
                target_height=TARGET_IMG_HEIGHT
            )
            
            print("\nValidating data sample against the suite...")
            results = validate_data_sample(EXPECTATION_SUITE_NAME, sample_df)
            
            if results:
                print(f"\nValidation Success: {results.success}")
                
                # Build and optionally open Data Docs for local inspection
                # Note: In CI, opening docs is not feasible, but building them can be useful for artifacts.
                try:
                    print("\nBuilding Data Docs...")
                    context.build_data_docs()
                    # Path to Data Docs index page (for local runs)
                    data_docs_path = os.path.join(
                        context.root_directory, "uncommitted", "data_docs", "local_site", "index.html"
                    )
                    print(f"Data Docs built. To view, open: {data_docs_path}")
                except Exception as e_docs:
                    print(f"Error building Data Docs: {e_docs}")
                
                if not results.success:
                    print("\nValidation Failures:")
                    for result in results.results:
                        if not result.success:
                            print(f"  - Expectation Type: {result.expectation_config.expectation_type}")
                            print(f"    Column: {result.expectation_config.kwargs.get('column')}")
                            print(f"    Details: {result.result}")
        else:
            print("No images loaded, skipping validation.")

    print("\nvalidate_data.py script finished.")
