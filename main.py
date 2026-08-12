from config.paths import DATASET_PATH
from Model.scripts.load_data import load_data
from Model.scripts.model_trainer import train_and_initialize_models


def main():
    train_data = load_data(DATASET_PATH)
    train_and_initialize_models(train_data)
    print("Training complete. Artifacts saved to /artifacts.")


if __name__ == "__main__":
    main()
