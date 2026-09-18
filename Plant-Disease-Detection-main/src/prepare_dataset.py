from pathlib import Path
import shutil
import random

# --------------------------------------------------
# Plant Disease Detection - Dataset Preparation
# --------------------------------------------------

SOURCE_DIR = Path("PlantVillage")
OUTPUT_DIR = Path("data/tomato")

CLASSES = {
    "Tomato___Bacterial_spot": "bacterial_spot",
    "Tomato___Early_blight": "early_blight",
    "Tomato___healthy": "healthy",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "spider_mites",
    "Tomato___Target_Spot": "target_spot",
    "Tomato___Tomato_mosaic_virus": "mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl_virus",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def find_class_folder(class_name):
    """Find a class folder even if the dataset naming differs slightly."""
    exact = SOURCE_DIR / class_name

    if exact.exists():
        return exact

    for folder in SOURCE_DIR.rglob("*"):
        if folder.is_dir() and folder.name.lower() == class_name.lower():
            return folder

    return None


def prepare_class(source_folder, output_name):

    images = [
        p for p in source_folder.rglob("*")
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    print(f"\n{output_name}")
    print(f"Total images : {total}")
    print(f"Train        : {len(splits['train'])}")
    print(f"Validation   : {len(splits['val'])}")
    print(f"Test         : {len(splits['test'])}")

    for split_name, split_images in splits.items():

        destination = OUTPUT_DIR / split_name / output_name
        destination.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(split_images):

            new_name = f"{output_name}_{index:05d}{image_path.suffix.lower()}"

            shutil.copy2(
                image_path,
                destination / new_name
            )


def main():

    if not SOURCE_DIR.exists():
        print("\nERROR:")
        print("PlantVillage folder was not found.")
        print("Place the downloaded dataset in the project root.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing tomato dataset...")
    print("=" * 50)

    for source_name, output_name in CLASSES.items():

        source_folder = find_class_folder(source_name)

        if source_folder is None:
            print(f"\nWARNING: {source_name} not found.")
            continue

        prepare_class(source_folder, output_name)

    print("\n" + "=" * 50)
    print("Dataset preparation completed.")


if __name__ == "__main__":
    main()