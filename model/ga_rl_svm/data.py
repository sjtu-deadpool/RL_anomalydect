import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define categories for attack types
Dos_type = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'processtable', 'udpstorm', 'apache2']
Probe_type = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
R2L_type = ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'xlock', 'xsnoop', 
            'snmpguess', 'snmpgetattack', 'sendmail', 'named', 'worm', 'warezclient']
U2R_type = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps', 'httptunnel']

# Function to classify attack types
def classify_type(x):
    if x in Dos_type:
        return 0
    elif x in Probe_type:
        return 0
    elif x in R2L_type:
        return 0
    elif x in U2R_type:
        return 0
    else:
        return 1  # Label for normal traffic

# Function to process and normalize the data
def process_data(data, rate=0):
    # Map categorical features in columns 1, 2, and 3 to numeric indices
    col_1_mapping = data[1].unique().tolist()
    col_2_mapping = data[2].unique().tolist()
    col_3_mapping = data[3].unique().tolist()
    
    # Drop the target level column (column 42)
    data = data.drop([42], axis=1)
    
    # Convert categorical features to numeric indices
    data[1] = data[1].apply(lambda x: col_1_mapping.index(x))
    data[2] = data[2].apply(lambda x: col_2_mapping.index(x))
    data[3] = data[3].apply(lambda x: col_3_mapping.index(x))
    
    # Map attack types to binary labels (0: attack, 1: normal)
    data[41] = data[41].apply(classify_type)
    
    # Separate features and labels
    labels = data[41]
    features = data.drop([41], axis=1)
    
    # If rate > 0, sample a portion of the dataset
    if rate > 0:
        features, _, labels, _ = train_test_split(features, labels, test_size=1 - rate)
    
    return features, labels

# Load and preprocess the training dataset
train_data = pd.read_csv('./nsl-kdd/KDDTrain+_20Percent.txt', header=None)
train_features, train_labels = process_data(train_data)

# Normalize training data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_features)
train_features = pd.DataFrame(scaler.transform(train_features))

# Load and preprocess the validation dataset
val_data = pd.read_csv('./nsl-kdd/KDDTest+.txt', header=None)
val_features, val_labels = process_data(val_data)

# Normalize validation data using MinMaxScaler
scaler.fit(val_features)
val_features = pd.DataFrame(scaler.transform(val_features))