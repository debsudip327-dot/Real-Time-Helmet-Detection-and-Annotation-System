import os
import shutil
import random

# ==============================
# SOURCE DATASET
# ==============================

SOURCE = os.path.join(os.getcwd(), "EdgeVision")

IMAGE_SOURCE = os.path.join(SOURCE, "images")
LABEL_SOURCE = os.path.join(SOURCE, "labels")

# ==============================
# OUTPUT DATASET
# ==============================

DEST = os.path.join(os.getcwd(), "helmet_dataset")

TRAIN_IMAGES = os.path.join(DEST, "images", "train")
VAL_IMAGES = os.path.join(DEST, "images", "val")

TRAIN_LABELS = os.path.join(DEST, "labels", "train")
VAL_LABELS = os.path.join(DEST, "labels", "val")

# ==============================
# CHECK SOURCE FOLDERS
# ==============================

if not os.path.exists(IMAGE_SOURCE):
    print("ERROR: Images folder not found!")
    print(IMAGE_SOURCE)
    exit()

if not os.path.exists(LABEL_SOURCE):
    print("ERROR: Labels folder not found!")
    print(LABEL_SOURCE)
    exit()

print("Source folders found successfully!")
print("Images:", IMAGE_SOURCE)
print("Labels:", LABEL_SOURCE)

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================

for folder in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS
]:
    os.makedirs(folder, exist_ok=True)

# ==============================
# FIND IMAGES
# ==============================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

images = []

for file in os.listdir(IMAGE_SOURCE):
    if file.lower().endswith(image_extensions):
        images.append(file)

print()
print("Total images found:", len(images))

# ==============================
# MATCH IMAGES WITH LABELS
# ==============================

valid_images = []
missing_labels = []

for image in images:

    base_name = os.path.splitext(image)[0]

    label_file = os.path.join(
        LABEL_SOURCE,
        base_name + ".txt"
    )

    if os.path.isfile(label_file):
        valid_images.append(image)
    else:
        missing_labels.append(image)

print("Images with labels:", len(valid_images))
print("Images without labels:", len(missing_labels))

# ==============================
# SHUFFLE DATA
# ==============================

random.seed(42)
random.shuffle(valid_images)

# ==============================
# TRAIN / VALIDATION SPLIT
# ==============================

split = int(len(valid_images) * 0.8)

train_images = valid_images[:split]
val_images = valid_images[split:]

print("Training images:", len(train_images))
print("Validation images:", len(val_images))

# ==============================
# COPY DATASET
# ==============================

def copy_dataset(image_list, image_destination, label_destination):

    for image in image_list:

        # Copy image
        source_image = os.path.join(
            IMAGE_SOURCE,
            image
        )

        destination_image = os.path.join(
            image_destination,
            image
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        # Copy label
        base_name = os.path.splitext(image)[0]

        source_label = os.path.join(
            LABEL_SOURCE,
            base_name + ".txt"
        )

        destination_label = os.path.join(
            label_destination,
            base_name + ".txt"
        )

        shutil.copy2(
            source_label,
            destination_label
        )

# ==============================
# COPY TRAINING DATA
# ==============================

print()
print("Copying training dataset...")

copy_dataset(
    train_images,
    TRAIN_IMAGES,
    TRAIN_LABELS
)

# ==============================
# COPY VALIDATION DATA
# ==============================

print("Copying validation dataset...")

copy_dataset(
    val_images,
    VAL_IMAGES,
    VAL_LABELS
)

# ==============================
# COMPLETE
# ==============================

print()
print("======================================")
print("   DATASET PREPARATION COMPLETE!")
print("======================================")
print()
print("Dataset location:")
print(DEST)
print()
print("Training images:", len(train_images))
print("Validation images:", len(val_images))
print("Images without labels:", len(missing_labels))