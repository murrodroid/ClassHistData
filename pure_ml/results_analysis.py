import pandas as pd
import os

def load_all_folds(folder_path, subfolders, folds=5):
    all_folds_data = {}
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        all_folds_data[subfolder] = []
        for i in range(1, folds+1):
            file_path = os.path.join(subfolder_path, f'cr_fold{i}.csv')
            df = pd.read_csv(file_path)
            all_folds_data[subfolder].append(df)
    return all_folds_data

def compute_averaged_per_label(df_list):
    combined_df = pd.concat(df_list, axis=0)
    if combined_df.columns[0] != 'label':
        combined_df = combined_df.rename(columns={combined_df.columns[0]: 'label'})
    filtered_df = combined_df[combined_df['support'] > 0]
    filtered_df = filtered_df[filtered_df['label'].apply(lambda x: str(x).isdigit())]
    filtered_df['label'] = filtered_df['label'].astype(int)
    averaged_df = filtered_df.groupby('label').mean(numeric_only=True).sort_index()
    return averaged_df

def compute_weighted_averages(averaged_df):
    total_support = averaged_df['support'].sum()
    weighted_precision = (averaged_df['precision'] * averaged_df['support']).sum() / total_support
    weighted_recall = (averaged_df['recall'] * averaged_df['support']).sum() / total_support
    weighted_f1 = (averaged_df['f1-score'] * averaged_df['support']).sum() / total_support
    return weighted_precision, weighted_recall, weighted_f1



if __name__ == '__main__':
    folder_path = 'trained_models/170325_retain_pct_0.5'
    subfolders = ['ordered_df_5Folds', 'random_df_5Folds']
    folds = 5

    all_folds_data = load_all_folds(folder_path, subfolders, folds)
    
    for subfolder in subfolders:
        averaged_df = compute_averaged_per_label(all_folds_data[subfolder])
        output_path = os.path.join(folder_path, f'{subfolder}_averaged_per_label.csv')
        averaged_df.to_csv(output_path)
        wp, wr, wf = compute_weighted_averages(averaged_df)
        print(f"Weighted averages for {subfolder}:")
        print(f"  Precision: {wp:.4f}")
        print(f"  Recall:    {wr:.4f}")
        print(f"  F1-Score:  {wf:.4f}")
