import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.utils import image_dataset_from_directory
import os

def load_images_from_directory(directory, target_size=(150, 150), batch_size=32):
    """
    Loads images from a directory, resizes them, prepares them in batches,
    and applies basic data augmentation.

    Args:
        directory (str): Path to the directory containing images.
                         It should have subdirectories for each class.
        target_size (tuple): Desired size for the images (height, width).
        batch_size (int): Number of images per batch.

    Returns:
        tf.data.Dataset: A TensorFlow dataset of augmented image batches.
                         Returns None if the directory does not exist or is empty.
    """
    if not os.path.exists(directory) or not os.listdir(directory):
        print(f"Error: Directory '{directory}' not found or is empty.")
        return None

    # Load images using image_dataset_from_directory
    dataset = image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',  # or 'categorical' or 'binary'
        image_size=target_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True
    )

    # Define data augmentation layers
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1), # Rotate by a factor of 0.1 (e.g. +/- 36 degrees if image is 360 degrees)
    ])

    # Apply data augmentation
    # AUTOTUNE allows TensorFlow to dynamically optimize the parallel calls
    # an AUTOTUNE value can be tf.data.AUTOTUNE
    dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch data for better performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

if __name__ == '__main__':
    # This is an example of how to use the function
    # Create dummy directories and images for testing
    if not os.path.exists("dummy_data/class_a"):
        os.makedirs("dummy_data/class_a")
    if not os.path.exists("dummy_data/class_b"):
        os.makedirs("dummy_data/class_b")

    # Create dummy image files (you might need to install Pillow: pip install Pillow)
    try:
        from PIL import Image
        # Create 5 dummy images for class_a
        for i in range(5):
            img = Image.new('RGB', (200, 200), color = 'red')
            img.save(f"dummy_data/class_a/img_a_{i}.png")
        # Create 5 dummy images for class_b
        for i in range(5):
            img = Image.new('RGB', (200, 200), color = 'blue')
            img.save(f"dummy_data/class_b/img_b_{i}.png")
        print("Dummy images created successfully.")

        # Example usage:
        img_height = 180
        img_width = 180
        batch_size_ = 3
        data_dir = "dummy_data"

        train_dataset = load_images_from_directory(data_dir,
                                                target_size=(img_height, img_width),
                                                batch_size=batch_size_)

        if train_dataset:
            print(f"\nDataset loaded successfully with {len(train_dataset)} batches.")
            print("Taking one batch to inspect...")
            for image_batch, labels_batch in train_dataset.take(1):
                print("Image batch shape:", image_batch.shape)
                print("Labels batch shape:", labels_batch.shape)
                print("First image in the batch (tensor):", image_batch[0])
                print("Label for the first image:", labels_batch[0])
        else:
            print("Failed to load the dataset.")

    except ImportError:
        print("Pillow library is not installed. Skipping dummy image creation and example usage.")
    except Exception as e:
        print(f"An error occurred during the example usage: {e}")

    finally:
        # Clean up dummy data
        if os.path.exists("dummy_data"):
            import shutil
            shutil.rmtree("dummy_data")
            print("\nCleaned up dummy data directory.")
