# Data augmentation parameters (only for training)
ROT_ANGLE = 5
W_SHIFT_RANGE = 0.05
H_SHIFT_RANGE = 0.05
FILL_MODE = "nearest"
BRIGHTNESS_RANGE = [0.95, 1.05]
VAL_SPLIT = 0.1

# Learning Rate Finder parameters
# funciona START_LR = 2e-5
START_LR = 2e-8
LR_MAX_EPOCHS = 5
LRF_DECREASE_FACTOR = 0.85

# Training parameters
#funciona EARLY_STOPPING = 20
EARLY_STOPPING = 50
REDUCE_ON_PLATEAU = 6

# Finetuning parameters
FINETUNE_SPLIT = 0.1
STEP_MIN_AREA = 5
START_MIN_AREA = 5
STOP_MIN_AREA = 1005
