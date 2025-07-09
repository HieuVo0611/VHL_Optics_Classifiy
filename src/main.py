import os

from config import DATA_DIR, META_COLORS
from processing import process_data
from normalize import getFeature
from model import train_models
from predict import predictImage

def e2e_pipeline() -> None:
    try:
        process_data(
            data_dir=DATA_DIR,
            meta_path=META_COLORS,
            folder_name='_uploadRGB_5phones_sorted',
        )
        print('\nProcessing Data Success!\n')
    except Exception as e:
        print(f'\nProcessing Data Failed: {e}\n')

    try:
        getFeature(
            df_path=META_COLORS,
            dir_path=os.path.join(DATA_DIR, 'square image'),
            out_path=os.path.join(DATA_DIR, 'csv')
        )
        print('\nFeature Extraction Success!\n')
    except Exception as e:
        print(f'\nFeature Extraction Failed: {e}\n')

    try:
        train_models(
            meta_path=META_COLORS,
            dir_path=os.path.join(DATA_DIR, 'csv'),
            out_path=os.path.join(DATA_DIR, 'models'),
            n_estimators=1000,
            n_splits=5
        )
        print('\nTraining Model Success!\n')
    except Exception as e:
        print(f'\nTraining Model Fail: {e}\n')

def predict_ui():
    out_path = os.path.join(DATA_DIR, 'predict')
    import pandas as pd
    try:
        df = pd.read_csv(META_COLORS)
        phones = sorted(df['Phones'].unique())
    except Exception as e:
        print(f"Failed to load phone list from metadata: {e}")
        return

    while True:
        choice = input("Do you want to predict an image? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                image_path = input('Enter the path to the image you want to predict: ').strip()

                print('========= Select your phone =========')
                for i, phone in enumerate(phones, 1):
                    print(f'{i}. {phone}')

                while True:
                    phone_input = input('Your selection: ').strip()
                    if phone_input.isdigit() and 1 <= int(phone_input) <= len(phones):
                        phone = phones[int(phone_input) - 1]
                        break
                    print('Invalid selection. Please try again.')

                print(f"Selected phone: {phone}\n")
                predictImage(
                    image_path=image_path,
                    out_path=out_path,
                    phone=phone,
                    summary_path=os.path.join(DATA_DIR, 'models', 'classification_summary.csv')
                )
                print("Prediction completed.\n")
            except Exception as e:
                print(f"Prediction Error: {e}")
        elif choice == 'n':
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")

if __name__ == '__main__':
    while True:
        choice = input("Do you want to run the end-to-end process? (y/n): ").strip().lower()
        if choice == 'y':
            e2e_pipeline()
            break
        elif choice == 'n':
            print("Skipping end-to-end process.")
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")

    predict_ui()
    