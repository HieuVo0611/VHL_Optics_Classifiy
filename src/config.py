import os

#Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists(os.path.join(ROOT_DIR, "data")):
    DATA_DIR = os.path.join(ROOT_DIR, "data")
else:
    os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
    DATA_DIR = os.path.join(ROOT_DIR, "data")

#Constants for data processing
COLUMNS = ['Id_imgs', 'Types', 'ppm', 'Phones', 'Num_of_photos', 'Date']
# COLUMNS_STD = ['Id_imgs', 'Types', 'Mean_B', 'Mean_G', 'Mean_R', 'Mode_B', 'Mode_G', 'Mode_R', 'Std_B', 'Std_G', 'Std_R']
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
META_COLORS = os.path.join(DATA_DIR,"metadata_colors.csv")


#Constants for image processing
SIZE_IMG = 224
SIZE_CUT = (300, 400)
RATIO = 0.75
HSV_KITS = {
    "1.1.1.1.0":[(35, 102, 76), (95, 255, 255)],    # Hue, Saturation, Value of 4 types of phones (mi8 lite, nokia, oppo, poco f3)
    "1.1.1.0.1":[(35, 100, 70), (100, 255, 255)]    # Hue, Saturation, Value of a phone (samsung)
    }

#CSV feature folder
CSV_DIR = os.path.join(DATA_DIR, 'csv')                    
CSV_ROI_DIR    = os.path.join(DATA_DIR, 'csv_roi')         # ROI
CSV_SQUARE_DIR = os.path.join(DATA_DIR, 'csv_square')      # Square

#Models result folder
MODELS_DIR = os.path.join(DATA_DIR, 'models')              
MODELS_ROI_DIR    = os.path.join(DATA_DIR, 'models_roi')    # ROI
MODELS_SQUARE_DIR = os.path.join(DATA_DIR, 'models_square') # Square

REPORT_DIR     = os.path.join(DATA_DIR, 'reports')
