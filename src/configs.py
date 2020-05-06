ON_SERVER = True
TRAIN_DIR = "../data/train/AOI_11_Rotterdam/"
TRAIN_SAR = f"{TRAIN_DIR}SAR-Intensity/"
TRAIN_RGB = f"{TRAIN_DIR}PS-RGB/"
TRAIN_JSON = f"{TRAIN_DIR}geojson_buildings/"
TRAIN_MASKS = f"{TRAIN_DIR}masks_np/"
TRAIN_META = f"{TRAIN_DIR}SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv"
TRAIN_FOLDS = f"{TRAIN_DIR}folds.csv"

TEST_DIR = "../../data/test_public/AOI_11_Rotterdam/"
TEST_SAR = f"{TEST_DIR}SAR-Intensity/"
RESULTS_DIR = "../output/"
LOSS = "cross_entropy"

NUM_CLASSES = 2
IMG_SIZE = 224


