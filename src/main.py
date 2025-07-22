import os

from config import DATA_DIR, META_COLORS, CSV_ROI_DIR, CSV_SQUARE_DIR, MODELS_ROI_DIR, MODELS_SQUARE_DIR
from processing import process_data
from normalize import getFeature
from model import train_models
from predict import predictImage
from visualize import generate_report

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
    
    # Pipineline ROI
    try:
        getFeature(
            df_path=META_COLORS,
            dir_path=os.path.join(DATA_DIR, 'roi image'),
            out_path=CSV_ROI_DIR,
            use_square_extras=False
        )
        print('\nFeature Extraction (ROI) Success!\n')
    
        train_models(
            meta_path=META_COLORS,
            dir_path=CSV_ROI_DIR,
            out_path=MODELS_ROI_DIR,
            n_estimators=1000,
            n_splits=5
        )
        print('\nTraining Model (ROI) Success!\n')
    except Exception as e:
        print(f'\nROI pipeline Failed: {e}\n')

    # Pipeline Square
    try:
        getFeature(
            df_path=META_COLORS,
            dir_path=os.path.join(DATA_DIR, 'square image'),
            out_path=CSV_SQUARE_DIR,
            use_square_extras=True
        )
        print('\nFeature Extraction (Square) Success!\n')
    
        train_models(
            meta_path=META_COLORS,
            dir_path=CSV_SQUARE_DIR,
            out_path=MODELS_SQUARE_DIR,
            n_estimators=1000,
            n_splits=5
        )
        print('\nTraining Model (Square) Success!\n')
    except Exception as e:
        print(f'\nSquare pipeline Failed: {e}\n')


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
            generate_report()
            break
        elif choice == 'n':
            print("Skipping end-to-end process.")
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")

    predict_ui()
    