from src.config import PROJECT_VERSION, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH
from src.versioning import dataset_version


def main() -> None:
    paths = [
        path
        for path in (TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH)
        if path.exists()
    ]
    print(f"project_version={PROJECT_VERSION}")
    if paths:
        print(f"dataset_version={dataset_version(paths)}")
    else:
        print("dataset_version=unavailable (run make split first)")


if __name__ == "__main__":
    main()
