import joblib
import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import random
import warnings

def load_and_prepare_model(model_path, X_test):
    """
    Load the model and prepare data for SHAP analysis.
    """
 
    loaded_pipeline = joblib.load(model_path)

    logistic_model = loaded_pipeline.named_steps['classifier']
    scaler = loaded_pipeline.named_steps['scaler']
    feature_selector = loaded_pipeline.named_steps['feature_selection']
    
    scaled_data = scaler.transform(X_test)
    
 
    if feature_selector.get_support().sum() < X_test.shape[1]:
        selected_features = X_test.columns[feature_selector.get_support()]
        scaled_samples = pd.DataFrame(
            scaled_data,
            columns=X_test.columns,
            index=X_test.index
        )[selected_features]
    else:
        scaled_samples = pd.DataFrame(
            scaled_data,
            columns=X_test.columns,
            index=X_test.index
        )
    
    return logistic_model, scaled_samples

def generate_shap_values(model, data, save_path=None):
    """
    Generate and optionally save SHAP values.
    """
    try:
        explainer = shap.KernelExplainer(
            model.predict_proba,
            shap.sample(data, 200)  # Use subset of data for background
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(data, nsamples=200)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if save_path:
            with open(save_path, 'wb') as file:
                pickle.dump(shap_values, file)
            print(f"SHAP values saved to {save_path}")
        
        return shap_values, explainer
        
    except Exception as e:
        print(f"Error generating SHAP values: {str(e)}")
        print("\nTrying alternative approach...")
        
        try:
    
            explainer = shap.LinearExplainer(
                model, 
                data,
                feature_dependence="correlation"
            )
            
            shap_values = explainer.shap_values(data)
    
            if len(shap_values.shape) > 1 and shap_values.shape[0] > 1:
                shap_values = shap_values[1]
            
            if save_path:
                with open(save_path, 'wb') as file:
                    pickle.dump(shap_values, file)
                print(f"SHAP values saved to {save_path}")
            
            return shap_values, explainer
            
        except Exception as e2:
            raise Exception(f"Both SHAP approaches failed.\nFirst error: {str(e)}\nSecond error: {str(e2)}")

def create_shap_plots(shap_values, data, output_dir):
    """
    Create and save various SHAP plots, showing only top 10 features for summary and beeswarm plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Calculate feature importance and get top 10 features
        feature_importance = np.abs(shap_values).mean(0)
        top_indices = np.argsort(-feature_importance)[:10]
        top_features = data.columns[top_indices]
        
        top_data = data.iloc[:, top_indices]
        top_shap_values = shap_values[:, top_indices]
        
        # 1. Summary plot 
        plt.figure(figsize=(12, 8))
        shap.summary_plot(top_shap_values, top_data, show=False)
        plt.title("SHAP Summary Plot (Top 10 Features)")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar plot 
        plt.figure(figsize=(12, 8))
        shap.summary_plot(top_shap_values, top_data, plot_type='bar', show=False)
        plt.title("SHAP Feature Importance (Top 10 Features)")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_importance_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Beeswarm plot 
        plt.figure(figsize=(12, 8))
        shap.summary_plot(top_shap_values, top_data, plot_type="dot", show=False)
        plt.title("SHAP Beeswarm Plot (Top 10 Features)")
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        #4. Top 10 feature dependency plots
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            feature_idx = np.where(data.columns == feature)[0][0]
            shap.dependence_plot(feature_idx, shap_values, data, show=False)
            plt.title(f"SHAP Dependence Plot: {feature}")
            plt.tight_layout()
            plt.savefig(output_dir / f'shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        specified_feature_pairs = [
                ('PDN_initially_post_Tx', 'Gender_0_F_1_M'),
                ('PDN_initially_post_Tx', 'Ageattransplantation'),
                ('PDN_initially_post_Tx', 'IgG_pre_Tx_ULN_A'),
                ('PDN_initially_post_Tx', 'Explant_Bx_necroinflammatory_activity'),
                ('PDN_1_y_post_Tx', 'Gender_0_F_1_M'),
                ('PDN_1_y_post_Tx', 'Ageattransplantation'),
                ('PDN_1_y_post_Tx', 'IgG_pre_Tx_ULN_A'),
                ('PDN_1_y_post_Tx', 'Explant_Bx_necroinflammatory_activity')
            ]
        data.columns = data.columns.str.strip()

        normalized_pairs = [
            (main.strip(), interaction.strip())
            for main, interaction in specified_feature_pairs
        ]

        for main_feature, interaction_feature in normalized_pairs:
            if main_feature not in data.columns or interaction_feature not in data.columns:
                print(f"Skipping: {main_feature} or {interaction_feature} not found in data.columns")
                continue

            plt.figure(figsize=(10, 6))
            main_feature_idx = np.where(data.columns == main_feature)[0][0]
            shap.dependence_plot(
                main_feature_idx,
                shap_values,
                data,
                interaction_index=interaction_feature,
                show=False
            )
            plt.title(f"SHAP Dependence Plot: {main_feature} vs {interaction_feature}")
            plt.tight_layout()
            plt.savefig(output_dir / f'shap_dependence_{main_feature}_vs_{interaction_feature}.png', dpi=300, bbox_inches='tight')
            plt.close()

            
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
        raise

def analyze_patient_examples(model, X_test, y_test, pipeline=None, explainer=None, shap_values=None, 
                           output_dir=None, n_samples=2, seed=123):
    """
    Analyze individual patient examples by selecting representative samples
    from positive and negative predicted classes.
    
    Parameters:
    -----------
    model : trained model
        The model to explain (final model in pipeline)
    X_test : pandas.DataFrame
        Test data features
    y_test : pandas.Series or numpy.ndarray
        True labels for test data
    pipeline : sklearn.pipeline.Pipeline, optional
        Full pipeline including preprocessing steps
    explainer : shap.Explainer, optional
        Pre-computed SHAP explainer
    shap_values : numpy.ndarray, optional
        Pre-computed SHAP values
    output_dir : str or Path, optional
        Directory to save visualizations
    n_samples : int, default=2
        Number of samples to select from each class
    seed : int, default=123
        Random seed for reproducibility
    """

    random.seed(seed)
    np.random.seed(seed)
    plt.rcParams['font.family'] = 'Times New Roman'
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions using the pipeline or model directly
    if pipeline is not None:
        X_processed = X_test.copy()
        for step_name, step in pipeline.named_steps.items():
            if step_name != 'classifier' and hasattr(step, 'transform') and not step_name.startswith('smote'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    X_processed = step.transform(X_processed)
        

        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_processed)
            y_pred = model.predict(X_processed)
            if y_pred_proba.shape[1] == 2:
                positive_proba = y_pred_proba[:, 1]
            else:
                positive_proba = y_pred_proba.max(axis=1)
        else:
            y_pred = model.predict(X_processed)
            positive_proba = y_pred
    else:
 
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            if y_pred_proba.shape[1] == 2:
                positive_proba = y_pred_proba[:, 1]
            else:
                positive_proba = y_pred_proba.max(axis=1)
        else:
            y_pred = model.predict(X_test)
            positive_proba = y_pred
    
    if pipeline and 'feature_selection' in pipeline.named_steps:
        feature_selector = pipeline.named_steps['feature_selection']
        if hasattr(feature_selector, 'get_support'):
            feature_mask = feature_selector.get_support()
            feature_names = X_test.columns[feature_mask].tolist()
        else:
            feature_names = X_test.columns.tolist()
    else:
        feature_names = X_test.columns.tolist()

    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    positive_indices = np.where(y_pred == 1)[0]
    negative_indices = np.where(y_pred == 0)[0]
    
    if len(positive_indices) < n_samples:
        print(f"Warning: Only {len(positive_indices)} positive samples found, using all of them.")
    if len(negative_indices) < n_samples:
        print(f"Warning: Only {len(negative_indices)} negative samples found, using all of them.")
    
    positive_samples = random.sample(list(positive_indices), min(n_samples, len(positive_indices))) if len(positive_indices) > 0 else []
    negative_samples = random.sample(list(negative_indices), min(n_samples, len(negative_indices))) if len(negative_indices) > 0 else []
    
    all_samples_indices = positive_samples + negative_samples

    # Prepare results dictionary
    results = {
        "positive_samples": [],
        "negative_samples": []
    }
    
    all_samples_df = pd.DataFrame()

    expected_value = 0
    if explainer is not None and hasattr(explainer, "expected_value"):
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            expected_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
        else:
            expected_value = explainer.expected_value
    
    # Analyze positive samples
    for i, idx in enumerate(positive_samples):
        raw_sample = X_test.iloc[idx]
        

        if shap_values is not None:
            sample_shap = shap_values[idx]
        else:
    
            print(f"Warning: No SHAP values provided for sample {idx}")
            continue
        
        # Store sample info
        sample_dict = {
            "index": idx,
            "features": raw_sample.to_dict(),
            "shap_values": sample_shap,
            "prediction": positive_proba[idx],
            "true_label": y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]
        }
    
        sample_df = pd.DataFrame([raw_sample.to_dict()], index=[idx])
        sample_df['predicted_probability'] = positive_proba[idx]
        sample_df['predicted_class'] = 'Positive'
        sample_df['true_label'] = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]
        sample_df['sample_type'] = 'Positive'

        all_samples_df = pd.concat([all_samples_df, sample_df])
        
        # Create force plot
        if output_dir:
            try:
                if len(feature_names) == len(sample_shap):
                 
                    plt.figure(figsize=(8,5))
                    plt.subplot()
                    
                    # Sort features by absolute SHAP value
                    feature_importance = list(zip(feature_names, sample_shap))
                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_10_features = feature_importance[:5]
                    top_10_features.reverse()

                    sorted_features = [item[0] for item in top_10_features]
                    sorted_values = [item[1] for item in top_10_features]
                    
                    y_pos = np.arange(len(sorted_features))
                    colors = ['#E14434' if x > 0 else '#5EABD6' for x in sorted_values]
                    
                    plt.barh(y_pos, sorted_values, color=colors)
                    plt.yticks(y_pos, sorted_features, fontsize=16)
                    
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                    plt.xlim(-0.04,0.08)

                    plt.xlabel('SHAP Value', fontsize=16)
                    plt.ylabel('Features', fontsize=16)
                    predicted_class = "Recurrence" if y_pred[idx] == 1 else "No Recurrence"
                    plt.title(f"Patient {idx} - Predicted Class: {predicted_class} (Probability: {positive_proba[idx]:.4f})", 
                            fontsize=16)
                    
                    red_patch = plt.Rectangle((0,0), 1, 1, fc="#E14434", edgecolor='none')
                    blue_patch = plt.Rectangle((0,0), 1, 1, fc="#5EABD6", edgecolor='none')
                    plt.legend([red_patch, blue_patch], ['Increases rAIH risk', 'Decreases rAIH risk'], 
                            loc='lower right')
     
                    plt.tight_layout()
                    plt.savefig(output_dir / f"positive_patient_{idx}_force_plot.png", 
                                dpi=600, bbox_inches='tight')
                    plt.close()
                else:
                    print(f"Warning: Feature length mismatch for sample {idx}. " 
                          f"Features: {len(feature_names)}, SHAP values: {len(sample_shap)}")
            except Exception as e:
                print(f"Error creating force plot for positive patient {idx}: {str(e)}")
        
        results["positive_samples"].append(sample_dict)
    
    # Analyze negative samples
    for i, idx in enumerate(negative_samples)
        raw_sample = X_test.iloc[idx]
        
        if shap_values is not None:
            sample_shap = shap_values[idx]
        else:
            print(f"Warning: No SHAP values provided for sample {idx}")
            continue
        
        sample_dict = {
            "index": idx,
            "features": raw_sample.to_dict(),
            "shap_values": sample_shap,
            "prediction": positive_proba[idx],
            "true_label": y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]
        }
        
        sample_df = pd.DataFrame([raw_sample.to_dict()], index=[idx])
        sample_df['predicted_probability'] = positive_proba[idx]
        sample_df['predicted_class'] = 'Negative'
        sample_df['true_label'] = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]
        sample_df['sample_type'] = 'Negative'
      
        all_samples_df = pd.concat([all_samples_df, sample_df])
    
        if output_dir:
            try:
                if len(feature_names) == len(sample_shap):
                    plt.figure(figsize=(8,5))
                    plt.subplot()

                    feature_importance = list(zip(feature_names, sample_shap))
                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_10_features = feature_importance[:5]
                    top_10_features.reverse()
                
                    sorted_features = [item[0] for item in top_10_features]
                    sorted_values = [item[1] for item in top_10_features]
                    
                    y_pos = np.arange(len(sorted_features))
                    colors = ['#E14434' if x > 0 else '#5EABD6' for x in sorted_values]
                    
                    plt.barh(y_pos, sorted_values, color=colors)
                    plt.yticks(y_pos, sorted_features, fontsize=16)
                
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                    plt.xlim(-0.04,0.08)
                    
                    plt.xlabel('SHAP Value', fontsize=16)
                    plt.ylabel('Features', fontsize=16)
                    predicted_class = "Recurrence" if y_pred[idx] == 1 else "No Recurrence"
                    plt.title(f"Patient {idx} - Predicted Class: {predicted_class} (Probability: {positive_proba[idx]:.4f})", 
                            fontsize=16)
            
                    red_patch = plt.Rectangle((0,0), 1, 1, fc="#E14434", edgecolor='none')
                    blue_patch = plt.Rectangle((0,0), 1, 1, fc="#5EABD6", edgecolor='none')
                    plt.legend([red_patch, blue_patch], ['Increases rAIH risk', 'DecreasesrAIH risk'], 
                            loc='lower right')

                    plt.tight_layout()
                    plt.savefig(output_dir / f"negative_patient_{idx}_force_plot.png", 
                                dpi=600, bbox_inches='tight')
                    plt.close()
                else:
                    print(f"Warning: Feature length mismatch for sample {idx}. " 
                          f"Features: {len(feature_names)}, SHAP values: {len(sample_shap)}")
            except Exception as e:
                print(f"Error creating force plot for negative patient {idx}: {str(e)}")
        
        results["negative_samples"].append(sample_dict)
    

    if output_dir and not all_samples_df.empty:
e
        shap_summary_df = pd.DataFrame(index=all_samples_df.index)
        for idx in all_samples_df.index:
          
            if idx in positive_indices:
                sample_list_idx = positive_samples.index(idx) if idx in positive_samples else -1
                if sample_list_idx >= 0:
                    sample_data = results["positive_samples"][sample_list_idx]
                    
                    if len(sample_data["shap_values"]) == len(feature_names):
                        feature_values = list(zip(feature_names, sample_data["shap_values"]))
                        feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
                      
                        for j, (feature, value) in enumerate(feature_values[:5]):
                            shap_summary_df.at[idx, f'top_{j+1}_feature'] = feature
                            shap_summary_df.at[idx, f'top_{j+1}_shap_value'] = value
                            shap_summary_df.at[idx, f'top_{j+1}_direction'] = "+" if value > 0 else "-"
            elif idx in negative_indices:
                sample_list_idx = negative_samples.index(idx) if idx in negative_samples else -1
                if sample_list_idx >= 0:
                    sample_data = results["negative_samples"][sample_list_idx]
                  
                    if len(sample_data["shap_values"]) == len(feature_names):
                        feature_values = list(zip(feature_names, sample_data["shap_values"]))
                        feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
                 
                        for j, (feature, value) in enumerate(feature_values[:5]):
                            shap_summary_df.at[idx, f'top_{j+1}_feature'] = feature
                            shap_summary_df.at[idx, f'top_{j+1}_shap_value'] = value
                            shap_summary_df.at[idx, f'top_{j+1}_direction'] = "+" if value > 0 else "-"

        all_samples_df.to_csv(output_dir / "patient_samples_features.csv")

        if not shap_summary_df.empty:
       
            merged_df = pd.merge(all_samples_df, shap_summary_df, left_index=True, right_index=True)
            merged_df.to_csv(output_dir / "patient_samples_with_shap.csv")
        
            shap_summary_df.to_csv(output_dir / "patient_samples_shap_summary.csv")
    print(f"Analysis complete. Selected {len(positive_samples)} positive and {len(negative_samples)} negative samples.")
    print(f"Saved patient features to {output_dir / 'patient_samples_features.csv'}")
    print(f"Saved patient features with SHAP summary to {output_dir / 'patient_samples_with_shap.csv'}")
    
    print("\nPositive class samples:")
    for i, sample in enumerate(results["positive_samples"]):
        print(f"  Sample {i+1} (Index {sample['index']}):")
        print(f"    Prediction: {sample['prediction']:.4f}")
        print(f"    True label: {sample['true_label']}")
        
        if len(sample["shap_values"]) == len(feature_names):
            feature_values = list(zip(feature_names, sample["shap_values"]))
            feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"    Top contributing features:")
            for feature, value in feature_values[:5]:
                direction = "+" if value > 0 else "-"
                print(f"      {feature}: {direction} {abs(value):.4f}")
        else:
            print("    Unable to determine top contributing features")
    
    print("\nNegative class samples:")
    for i, sample in enumerate(results["negative_samples"]):
        print(f"  Sample {i+1} (Index {sample['index']}):")
        print(f"    Prediction: {sample['prediction']:.4f}")
        print(f"    True label: {sample['true_label']}")

        if len(sample["shap_values"]) == len(feature_names):
            feature_values = list(zip(feature_names, sample["shap_values"]))
            feature_values.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"    Top contributing features:")
            for feature, value in feature_values[:5]:
                direction = "+" if value > 0 else "-"
                print(f"      {feature}: {direction} {abs(value):.4f}")
        else:
            print("    Unable to determine top contributing features")
    
    return results, all_samples_df
    
def main(dataset_name, X_test, model_path, shap_save_path, output_dir):
    """
    Main function for SHAP analysis. It loads the model, generates SHAP values,
    and creates the SHAP plots for the given dataset (either 'all' or 'adults').
    """
    # Load model and prepare data
    logistic_model, scaled_samples = load_and_prepare_model(model_path, X_test)
    
    # Generate SHAP values
    shap_values, explainer = generate_shap_values(
        logistic_model, 
        scaled_samples,
        save_path=shap_save_path
    )
    
    # Create SHAP plots
    create_shap_plots(shap_values, scaled_samples, output_dir)
    
    print(f"SHAP analysis completed for {dataset_name}. Plots saved in:", output_dir)
    
    return shap_values, explainer, scaled_samples


def load_train_test_data(df, dataset):
    """
    Load the train and test data for the given dataset.
    """
    px_ids = {}
    
    for split in ["train", "test"]:
        with open(f"/Users/iriesun/MLcodes/AIH/data/data_splits_{dataset}/{split}_split.txt") as f:
            px_ids[split] = [float(id) for id in f]

    train_df = df[df["rAIHID"].isin(px_ids["train"])]
    test_df = df[df["rAIHID"].isin(px_ids["test"])]
    cols = list(train_df.columns)

    return train_df, test_df, cols


if __name__ == "__main__":
    # Load datasets
    df_all = pd.read_csv('/AIH_recurrence_ML/data/AIH_processed_all.csv', index_col=0)
 
    common_columns = ['rAIHID', 'rec', 'Gender_0_F_1_M', 'Ethnicity4Groups_0.0',
       'Ethnicity4Groups_1.0', 'Ethnicity4Groups_2.0', 'Ethnicity4Groups_3.0',
       'Ageatdiagnosis', 'Ageattransplantation', 'AIH_type', 'Overlap',
       'Concomitant_autoimmune_diseases', 'ANA_pre_Tx', 'ASMA_pre_Tx',
       'AMA_pre_Tx', 'ABO_recipient_A', 'ABO_recipient_AB', 'ABO_recipient_B',
       'ABO_recipient_O', 'Budesonide_pre_Tx', 'Esophageal_varices_pre_Tx',
       'Variceal_hemorrhage_pre_Tx', 'Hepatic_encephalopathy_pre_Tx',
       'Ascites_pre_Tx', 'TxmonthsafterDx','Explant_Bx_necroinflammatory_activity',
       'Donor_age', 'Donor_gender', 'GenderMismatch', 'IgG_pre_Tx_ULN_A',
       'IgA_pre_Tx_ULN', 'IgM_pre_Tx_ULN', 'Rejection', 'Sepsis_post_Tx',
       'ALT_pre_Tx', 'AST_pre_Tx', 'ALP_pre_Tx', 'Bilirubin_pre_Tx',
       'INR_pre_Tx', 'Creatitine_pre_Tx', 'MELD_pre_Tx', 'ALT_3_m', 'AST_3',
       'ALP_3_m', 'Bilirubin_3_m', 'ALT_6_m', 'AST_6_m', 'ALP_6_m',
       'Bilirubin_6_m', 'ALT_1_y', 'AST_1_y', 'ALP_1_y', 'Bilirubin_1_y',
       'Prednisone_pre_Tx', 'AZA_pre_Tx', 'MMF_pre_Tx',
       'Tacrolimus_initially_post_Tx', 'Cyclosporine_initially_post_Tx',
       'mTOR_inhibitors_initially_post_Tx', 'MMF_AZA', 'PDN_initially_post_Tx',
       'PDN_1_y_post_Tx', 'imm_regimen_initial', 'imm_regimen_1y',
       'initial_binary_cyc_trough', 'initial_binary_tac_trough',
       'initial_binary_mTOR_trough', '1y_binary_cyc_trough',
       '1y_binary_tac_trough', '1y_binary_mTOR_trough']
    

    df_all = df_all[common_columns]

    train_df_all, test_df_all, cols_all = load_train_test_data(df_all, 'all') 
    X_test_all = test_df_all.drop(['rec', 'rAIHID'], axis=1)
    y_test_all = test_df_all['rec']
    
    
    # Load models
    model_path_all = '/AIH_recurrence_ML/models/logistic_regression.joblib'

    shap_save_path_all = 'AIH_shap_values_all_logreg.pkl'

    output_dir_all = "/AIH_recurrence_ML/models/shap_analysis/patient_examples_all"
    
    shap_values_all, explainer_all, scaled_samples_all = main('all', X_test_all, model_path_all, shap_save_path_all, output_dir_all)

    loaded_pipeline_all = joblib.load(model_path_all)
    model_all = loaded_pipeline_all.named_steps['classifier']


    patient_examples_all = analyze_patient_examples(
        model=model_all,
        X_test=X_test_all,
        y_test=y_test_all,
        pipeline=loaded_pipeline_all,  # Pass the full pipeline
        explainer=explainer_all,
        shap_values=shap_values_all,
        output_dir=output_dir_all,
        n_samples=3  # Analyze 3 patients from each class
    )

