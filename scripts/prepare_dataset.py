import os
import shutil
from sklearn.model_selection import train_test_split


def prepare_dataset(
        dataset_path="data/face-expression-recognition-dataset/images/images",
        split_ratios=(0.7, 0.15, 0.15)
):
    """
    1. Combines 'train' & 'validation' into 'combined'.
    2. Splits into new train, validation, and test sets by the given ratios.
    3. Deletes 'combined' directory to avoid confusion.
    """
    print("===> Preparing dataset with custom splits...")

    train_source = os.path.join(dataset_path, 'train')
    val_source = os.path.join(dataset_path, 'validation')
    combined_dataset = os.path.join(dataset_path, 'combined')

    # Step 1: Combine old train & validation
    if not os.path.exists(train_source) or not os.path.exists(val_source):
        print("[ERROR] Could not find original 'train' or 'validation' directories.")
        return

    if not os.path.exists(combined_dataset):
        os.makedirs(combined_dataset)
        # List emotion categories
        emotions = [d for d in os.listdir(train_source)
                    if os.path.isdir(os.path.join(train_source, d)) and not d.startswith('.')]
        print(f"Combining the following emotion classes: {emotions}")

        for emotion in emotions:
            emotion_combined_dir = os.path.join(combined_dataset, emotion)
            os.makedirs(emotion_combined_dir, exist_ok=True)

            # Copy training images
            train_emotion_dir = os.path.join(train_source, emotion)
            for img in os.listdir(train_emotion_dir):
                if not img.startswith('.'):
                    shutil.copy(os.path.join(train_emotion_dir, img), emotion_combined_dir)

            # Copy validation images
            val_emotion_dir = os.path.join(val_source, emotion)
            for img in os.listdir(val_emotion_dir):
                if not img.startswith('.'):
                    shutil.copy(os.path.join(val_emotion_dir, img), emotion_combined_dir)

    # Step 2: Create new split
    emotions = [d for d in os.listdir(combined_dataset)
                if os.path.isdir(os.path.join(combined_dataset, d)) and not d.startswith('.')]

    new_train_dir = os.path.join(dataset_path, 'new_train')
    new_val_dir = os.path.join(dataset_path, 'new_validation')
    new_test_dir = os.path.join(dataset_path, 'test')

    for d in [new_train_dir, new_val_dir, new_test_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    print(f"Splitting data: {int(split_ratios[0] * 100)}% train, "
          f"{int(split_ratios[1] * 100)}% val, {int(split_ratios[2] * 100)}% test.\n")

    for emotion in emotions:
        emotion_dir = os.path.join(combined_dataset, emotion)
        images = [
            img for img in os.listdir(emotion_dir)
            if os.path.isfile(os.path.join(emotion_dir, img)) and not img.startswith('.')
        ]

        if len(images) < 3:
            print(f"[WARNING] Not enough images in '{emotion}' to split properly.")
            continue

        train_imgs, temp_imgs = train_test_split(
            images,
            test_size=(1 - split_ratios[0]),
            random_state=42
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=(split_ratios[2] / (split_ratios[1] + split_ratios[2])),
            random_state=42
        )

        # Move images into new dirs
        for imgs, subset_dir in zip([train_imgs, val_imgs, test_imgs],
                                    [new_train_dir, new_val_dir, new_test_dir]):
            emotion_subset_dir = os.path.join(subset_dir, emotion)
            os.makedirs(emotion_subset_dir, exist_ok=True)
            for img in imgs:
                shutil.move(os.path.join(emotion_dir, img), os.path.join(emotion_subset_dir, img))

    # Step 3: Clean up
    shutil.rmtree(combined_dataset)
    print("===> Dataset preparation complete!")
    print(f"New Train Dir: {new_train_dir}")
    print(f"New Val Dir:   {new_val_dir}")
    print(f"New Test Dir:  {new_test_dir}\n")
    print("You can now use 'new_train', 'new_validation', and 'test' for your experiments.")


if __name__ == "__main__":
    prepare_dataset()
