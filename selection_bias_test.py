# packages
import os
from sklearn.preprocessing import StandardScaler
import datetime

# modules
from modules.import_data import import_data_random
from modules.text_preprocessor import *
from modules.tools import return_device, undersampling
from modules.model_evaluation import validate_models
from modules.training import train_k_folds
from modules.networks import individualized_network
from modules.tools import save_hyper_parameters



#### hyper-parameters
learning_rate = 0.0008
batch_size = 64
num_epochs = 2
dropout_rate = 0.6
retain_pct = 0.4
k_folds = 5
undersampling_scale = 0.4
####


date = datetime.datetime.now().strftime('%d%m%y')

device = return_device()

df, _ = import_data_random(retain_pct)
print('Data imported successfully.')

# undersampling might be a huge bias - talk with Mads
df = undersampling(df=df, target_col='icd10h', scale=undersampling_scale, lower_bound=50)

model_names = ['random_df','ordered_df']
random_df, val_random_df = df[df.icd10h_random.notna()], df[df.icd10h_random.isna()]
ordered_df, val_ordered_df = df[df.icd10h_ordered.notna()], df[df.icd10h_ordered.isna()]

token_types = [
    {'method': 'char', 'ngram': 2},
    {'method': 'char', 'ngram': 3},
    {'method': 'word', 'ngram': 0}
]


for i, train_df in enumerate([random_df, ordered_df]):
    print(f'K-fold training & evaluation of model {model_names[i]} initiated.')

    model_folder = f'trained_models/{date}_retain_pct_{retain_pct}/{model_names[i]}_{k_folds}Folds'
    os.makedirs(model_folder, exist_ok=True)
    
    print(f'Total size of dataset: {train_df.shape}')

    scaler_age = StandardScaler().fit(train_df['age'].to_numpy().reshape(-1, 1))
    le_sex = LabelEncoder().fit(train_df['sex'])
    
    X_cause, vocab = prepare_deathcauses_tensors(df=train_df, column='deathcause_mono', token_types=token_types)
    X_age = torch.tensor(scaler_age.transform(train_df['age'].to_numpy().reshape(-1, 1)), dtype=torch.float).to(device)
    X_sex = torch.tensor(le_sex.transform(train_df['sex']), dtype=torch.long).to(device)

    y_column = 'icd10h_random' if i == 0 else 'icd10h_ordered'
    y_tensor, y_label_encoder = encode_labels(train_df, transform_column=y_column, fit_df=df, fit_column='icd10h')
    num_classes = len(y_label_encoder.classes_)
    criterion = nn.CrossEntropyLoss()


    fold_model_paths = train_k_folds(i,X_cause,X_age,X_sex,y_tensor,model_names,model_folder,vocab,num_classes,dropout_rate,learning_rate,batch_size,num_epochs,individualized_network,k_folds,criterion,device)
    save_hyper_parameters(model_folder,f"{model_names[i]}_hyper_parameters.txt",dropout_rate,learning_rate,num_epochs,retain_pct,undersampling_scale)
    
    val_df = val_random_df if i == 0 else val_ordered_df
    validate_models(val_df, fold_model_paths, scaler_age, le_sex, y_label_encoder, token_types, batch_size, num_classes, vocab, dropout_rate, criterion, model_folder, device)