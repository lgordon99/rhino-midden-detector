'''utils.py by Lucia Gordon'''

# imports
import numpy as np
import yaml

# functions
def get_project_dir():
    '''get path to the project directory'''
    with open('code/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    return config['project_dir']

def get_site():
    '''get path to the site being used'''
    with open('code/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    return config['site']

def str_to_bool(string):
    '''convert a string input to a Boolean variable'''
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    return False

def get_constants():
    return np.load(f'{get_project_dir()}/{get_site()}/data/constants.npy')

def get_thermal_interval():
    return int(get_constants()[0][1])

def get_thermal_stride():
    return int(get_constants()[1][1])

def get_thermal_left():
    return float(get_constants()[2][1])

def get_thermal_top():
    return float(get_constants()[3][1])

def get_thermal_pixel_width():
    return float(get_constants()[4][1])

def get_thermal_pixel_height():
    return float(get_constants()[5][1])

def get_num_horizontal():
    return int(get_constants()[8][1])

def get_image_center_pixels(identifier):
    NUM_HORIZONTAL = get_num_horizontal()
    THERMAL_STRIDE = get_thermal_stride()
    THERMAL_INTERVAL = get_thermal_interval()

    row = np.floor(identifier/NUM_HORIZONTAL)
    col = identifier - NUM_HORIZONTAL*np.floor(identifier/NUM_HORIZONTAL)
    x_pixels = col*(THERMAL_STRIDE+THERMAL_INTERVAL/2) + THERMAL_INTERVAL/2
    y_pixels = row*(THERMAL_STRIDE+THERMAL_INTERVAL/2) + THERMAL_INTERVAL/2

    return x_pixels, y_pixels

def get_image_center_meters(x_pixels, y_pixels):
    THERMAL_LEFT = get_thermal_left()
    THERMAL_TOP = get_thermal_top()
    THERMAL_PIXEL_WIDTH = get_thermal_pixel_width()
    THERMAL_PIXEL_HEIGHT = get_thermal_pixel_height()
    
    x = THERMAL_LEFT + x_pixels*THERMAL_PIXEL_WIDTH
    y = THERMAL_TOP + y_pixels*THERMAL_PIXEL_HEIGHT

    return x, y

# def pad_array(array, pad_value, row_divisor, column_divisor):
#     '''Pads an array by adding rows and/or columns filled with a constant value to the bottom and/or right of the array'''

#     num_input_rows, num_input_cols = array.shape
#     num_rows_to_append = int((row_divisor - num_input_rows%row_divisor) % row_divisor) # divisible by row divisor
#     num_cols_to_append = int((column_divisor - num_input_cols%column_divisor) % column_divisor) # divisible by column divisor
#     rows_to_append = np.full((num_rows_to_append, num_input_cols), pad_value)
#     padded_array = np.vstack((array, rows_to_append)) # adds rows to the bottom
#     cols_to_append = np.full((padded_array.shape[0], num_cols_to_append), pad_value)
#     padded_array = np.hstack((padded_array, cols_to_append)) # adds columns to the right

#     return padded_array

# def calculate_percentile(array, percentile):
#     '''Calculates the percentile of the non-NaN values of an array according to the linear interpolation method'''

#     array = np.array(array) # converts array to a numpy array if it is not already
#     array_without_nans = array[array != NAN] # 1D array of non-NaN values

#     if len(array_without_nans) == 0: # if the original array contains all NaNs
#         return NAN

#     sorted_array = sorted(array_without_nans)
#     index = percentile/100 * (len(sorted_array) - 1)

#     if index.is_integer():
#         return sorted_array[int(index)]

#     # linear interpolation between the lower and upper indices is performed if the index is not an integer
#     lower_index = int(index)
#     upper_index = lower_index + 1
#     interpolation = index - lower_index
#     percentile = sorted_array[lower_index] + interpolation * (sorted_array[upper_index] - sorted_array[lower_index])

#     return percentile
