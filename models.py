import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, make_scorer, recall_score, roc_auc_score, f1_score, fbeta_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
from pathlib import Path
import datetime
import warnings
warnings.filterwarnings('ignore')

def specificity_score(y_true, y_pred):
    """Calculate specificity = TN / (TN + FP)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def balanced_score(y_true, y_pred):
    """Calculate balanced score as (sensitivity + specificity) / 2."""
    sensitivity = recall_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    return (sensitivity + spec) / 2

def create_voting_classifier(estimators, weights=None):
    """Create a voting classifier with optional weights."""
    return VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights,
        n_jobs=-1
    )

def format_metric_with_std(mean, std):
    """Format mean and standard deviation as string."""
    return f"{mean:.3f} ± {std:.3f}"

def save_best_models(results, best_estimators, dataset, save_dir='best_models'):
    """Save the best performing models and their metadata."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(f'/AIH/models/saved_models/cw_stacking/{dataset}/{save_dir}_{timestamp}')
    save_path.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(save_path / 'model_results.csv')
    
    for name, model in best_estimators.items():
        model_path = save_path / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
    
    print(f"\nModels saved in: {save_path}")
    return save_path

def bootstrap_performance(model, X_test, y_test, n_bootstraps=1000, random_state=123):
    """
    Calculate model performance metrics using bootstrapping.
    
    Parameters:
    -----------
    model : trained sklearn model
        The trained model to evaluate
    X_test : pandas DataFrame
        Test features
    y_test : pandas Series
        Test target values
    n_bootstraps : int, default=1000
        Number of bootstrap samples
    random_state : int, default=123
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary of performance metrics with confidence intervals
    """
    np.random.seed(random_state)
    n_samples = X_test.shape[0]
    
    auroc = np.zeros(n_bootstraps)
    sensitivity = np.zeros(n_bootstraps)
    specificity = np.zeros(n_bootstraps)
    balanced = np.zeros(n_bootstraps)
    f1 = np.zeros(n_bootstraps)
    precision = np.zeros(n_bootstraps)
    
    bootstrap_indices = np.random.randint(0, n_samples, (n_bootstraps, n_samples))
    
    for i in range(n_bootstraps):
        indices = bootstrap_indices[i]
        
        X_bootstrap = X_test.iloc[indices]
        y_bootstrap = y_test.iloc[indices]
        
        y_pred = model.predict(X_bootstrap)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_bootstrap)[:, 1]
            auroc[i] = roc_auc_score(y_bootstrap, y_proba)
        else:
            auroc[i] = 0  

        sensitivity[i] = recall_score(y_bootstrap, y_pred)
        specificity[i] = specificity_score(y_bootstrap, y_pred)
        balanced[i] = balanced_score(y_bootstrap, y_pred)
        f1[i] = f1_score(y_bootstrap, y_pred)
        precision[i] = precision_score(y_bootstrap, y_pred)
    
    def get_ci(metric):
        lower = np.percentile(metric, 2.5)
        upper = np.percentile(metric, 97.5)
        mean = np.mean(metric)
        return mean, (upper - lower) / 2
    
    results = {
        'AUROC': get_ci(auroc),
        'Sensitivity': get_ci(sensitivity),
        'Specificity': get_ci(specificity),
        'Balanced Score': get_ci(balanced),
        'F1 Score': get_ci(f1),
        'Precision': get_ci(precision)
    }
    
    return results

def train_models_with_class_weights_and_voting(X_train, y_train, beta=2):
    """
    Train multiple binary classification models using class weights for class imbalance.
    Returns all models plus optimized voting and stacking classifiers.
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series
        Training target values
    beta : float, default=2
        Beta parameter for F-beta score
        
    Returns:
    --------
    tuple : (dictionary of trained models, metadata dictionary)
    """
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   
    fbeta_scorer = make_scorer(fbeta_score, beta=beta)
    scoring = {
        'auroc': 'roc_auc',
        'sensitivity': make_scorer(recall_score),
        'specificity': make_scorer(specificity_score),
        'balanced_score': make_scorer(balanced_score),
        'f1': 'f1',
        f'fbeta_{beta}': fbeta_scorer
    }
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    pos_neg_ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])

    
    print(f"Class weights: {class_weights}")
    print(f"Positive to negative ratio: {pos_neg_ratio:.2f}")
    
    # Logistic Regression
    print("\nTraining Logistic Regression...")
    logreg = LogisticRegression(class_weight=class_weight_dict, random_state=123, n_jobs=-1)
    logreg_param_grid = {
        'logisticregression__C': [0.01, 0.03, 0.05, 0.1]
    }
    logreg_pipeline = ImbPipeline([('scaler', StandardScaler()), ('logisticregression', logreg)])
    logreg_grid_search = GridSearchCV(logreg_pipeline, logreg_param_grid, cv=cv_folds, scoring='recall', n_jobs=-1)
    logreg_grid_search.fit(X_train, y_train)
    logreg_model = logreg_grid_search.best_estimator_
    print("Best Logistic Regression Hyperparameters:", logreg_grid_search.best_params_)

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(class_weight=class_weight_dict, random_state=123, n_jobs=-1)
    rf_param_grid = {
        'randomforestclassifier__n_estimators': [16, 32, 64],
        'randomforestclassifier__max_depth': [4, 8, 16],
        'randomforestclassifier__min_samples_leaf': [1, 2, 4]
    }
    rf_pipeline = ImbPipeline([('randomforestclassifier', rf)])
    rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv_folds, scoring='recall', n_jobs=-1)
    rf_grid_search.fit(X_train, y_train)
    rf_model = rf_grid_search.best_estimator_
    print("Best Random Forest Hyperparameters:", rf_grid_search.best_params_)

    # XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(eval_metric='logloss', scale_pos_weight=pos_neg_ratio, random_state=123, n_jobs=-1)
    xgb_param_grid = {
        'xgbclassifier__n_estimators': [16, 32, 64],
        'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
        'xgbclassifier__max_depth': [4, 8, 16]
    }
    xgb_pipeline = ImbPipeline([('xgbclassifier', xgb)])
    xgb_grid_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=cv_folds, scoring='recall', n_jobs=-1)
    xgb_grid_search.fit(X_train, y_train)
    xgb_model = xgb_grid_search.best_estimator_
    print("Best XGBoost Hyperparameters:", xgb_grid_search.best_params_)

    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingClassifier(loss='log_loss', random_state=123)
    gb_param_grid = {
        'gb__n_estimators': [16, 32, 64],
        'gb__learning_rate': [0.01, 0.1, 0.2],
        'gb__max_depth': [4, 8, 16]
    }
    gb_pipeline = ImbPipeline([('gb', gb)])
    gb_grid_search = GridSearchCV(gb_pipeline, gb_param_grid, cv=cv_folds, scoring='recall', n_jobs=-1)
    gb_grid_search.fit(X_train, y_train)
    gb_model = gb_grid_search.best_estimator_
    print("Best Gradient Boosting Hyperparameters:", gb_grid_search.best_params_)

    # SVM
    print("\nTraining SVM...")
    svm = SVC(probability=True, class_weight=class_weight_dict, random_state=123)
    svm_param_grid = {
        'svm__C': [0.01, 0.1, 1],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto']
    }
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', svm)
    ])
    svm_grid_search = GridSearchCV(svm_pipeline, svm_param_grid, cv=cv_folds, scoring='recall', n_jobs=-1)
    svm_grid_search.fit(X_train, y_train)
    svm_model = svm_grid_search.best_estimator_
    print("Best SVM Hyperparameters:", svm_grid_search.best_params_)

    # Store all models in a dictionary
    models = {
        'Logistic Regression': logreg_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'Gradient Boosting': gb_model,
        'SVM': svm_model
    }
    
    # Evaluate each model to find highest sensitivity and specificity
    sensitivity_scores = {}
    specificity_scores = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name} using cross-validation...")
        y_pred = cross_val_predict(model, X_train, y_train, cv=cv_folds)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        sensitivity_scores[name] = sensitivity
        specificity_scores[name] = specificity
        
        print(f"{name} - Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    
    # Find models with highest sensitivity and specificity
    best_sensitivity_model_name = max(sensitivity_scores, key=sensitivity_scores.get)
    best_specificity_model_name = max(specificity_scores, key=specificity_scores.get)
    
    best_sensitivity_model = models[best_sensitivity_model_name]
    best_specificity_model = models[best_specificity_model_name]
    
    print(f"\nBest model for sensitivity: {best_sensitivity_model_name} with score: {sensitivity_scores[best_sensitivity_model_name]:.4f}")
    print(f"Best model for specificity: {best_specificity_model_name} with score: {specificity_scores[best_specificity_model_name]:.4f}")
    
    # Optimized voting classifier with the two best models plus logistic regression
    print("\nCreating optimized voting classifier...")
    optimized_voting_clf = VotingClassifier(
        estimators=[
            ('logreg', logreg_model),
            ('sensitivity', best_sensitivity_model),
            ('specificity', best_specificity_model)
        ],
        voting='soft',
        weights=[1,1,1]  
    )
    optimized_voting_clf.fit(X_train, y_train)
    
    # Original voting classifier with all models
    print("\nCreating original voting classifier...")
    original_voting_clf = VotingClassifier(
        estimators=[
            ('logreg', logreg_model),
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('svm', svm_model)
        ],
        voting='soft',
        weights=[1, 1, 1, 1]
    )
    original_voting_clf.fit(X_train, y_train)
    
    # Stacking classifier
    print("\nCreating stacking classifier...")
    meta_learner = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        max_iter=2000,
        random_state=123
    )
    stacking_clf = StackingClassifier(
        estimators=[
            ('logreg', logreg_model),
            ('sensitivity', best_sensitivity_model),
            ('specificity', best_specificity_model)
        ],
        final_estimator=meta_learner,
        cv=cv_folds,
        passthrough=True
    )
    stacking_clf.fit(X_train, y_train)

    # Add ensemble models to the dictionary
    models['Optimized Voting Classifier'] = optimized_voting_clf
    models['Original Voting Classifier'] = original_voting_clf
    models['Stacking Classifier'] = stacking_clf

    # Collect and format cross-validation results
    results = {}
    for name, model in models.items():
        print(f"\nCollecting cross-validation metrics for {name}...")
        try:
            scores = cross_validate(
                model,
                X_train, y_train,
                scoring=scoring,
                cv=cv_folds,
                n_jobs=-1
            )
            
            results[name] = {
                'auroc_mean': np.mean(scores['test_auroc']),
                'auroc_std': np.std(scores['test_auroc']),
                'sensitivity_mean': np.mean(scores['test_sensitivity']),
                'sensitivity_std': np.std(scores['test_sensitivity']),
                'specificity_mean': np.mean(scores['test_specificity']),
                'specificity_std': np.std(scores['test_specificity']),
                'balanced_score_mean': np.mean(scores['test_balanced_score']),
                'balanced_score_std': np.std(scores['test_balanced_score']),
                'f1_mean': np.mean(scores['test_f1']),
                'f1_std': np.std(scores['test_f1']),
                f'fbeta_{beta}_mean': np.mean(scores[f'test_fbeta_{beta}']),
                f'fbeta_{beta}_std': np.std(scores[f'test_fbeta_{beta}'])
            }
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue

    # Format results for display
    formatted_results = {}
    for model_name, metrics in results.items():
        formatted_results[model_name] = {
            'AUROC': format_metric_with_std(metrics['auroc_mean'], metrics['auroc_std']),
            'Sensitivity': format_metric_with_std(metrics['sensitivity_mean'], metrics['sensitivity_std']),
            'Specificity': format_metric_with_std(metrics['specificity_mean'], metrics['specificity_std']),
            'Balanced Score': format_metric_with_std(metrics['balanced_score_mean'], metrics['balanced_score_std']),
            'F1 Score': format_metric_with_std(metrics['f1_mean'], metrics['f1_std']),
            f'F{beta} Score': format_metric_with_std(metrics[f'fbeta_{beta}_mean'], metrics[f'fbeta_{beta}_std'])
        }

    metadata = {
        'best_sensitivity_model': best_sensitivity_model_name,
        'best_specificity_model': best_specificity_model_name,
        'class_weights': class_weight_dict,
        'pos_neg_ratio': pos_neg_ratio
    }
    
    return models, metadata, pd.DataFrame(formatted_results).T

def evaluate_models_with_bootstrap(models, X_test, y_test, beta=2):
    """
    Evaluate models on test data using bootstrapping.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pandas DataFrame
        Test features
    y_test : pandas Series
        Test target values
    beta : float, default=2
        Beta parameter for F-beta score
        
    Returns:
    --------
    pandas DataFrame : Results dataframe
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name} on test set with bootstrapping...")
        
        try:
            bootstrap_results = bootstrap_performance(model, X_test, y_test)
            
            # Calculate F-beta manually since it's not in the bootstrap_performance function
            y_pred = model.predict(X_test)
            fbeta = fbeta_score(y_test, y_pred, beta=beta)
            
            results[name] = {
                'AUROC': format_metric_with_std(bootstrap_results['AUROC'][0], bootstrap_results['AUROC'][1]),
                'Sensitivity': format_metric_with_std(bootstrap_results['Sensitivity'][0], bootstrap_results['Sensitivity'][1]),
                'Specificity': format_metric_with_std(bootstrap_results['Specificity'][0], bootstrap_results['Specificity'][1]),
                'Balanced Score': format_metric_with_std(bootstrap_results['Balanced Score'][0], bootstrap_results['Balanced Score'][1]),
                'F1 Score': format_metric_with_std(bootstrap_results['F1 Score'][0], bootstrap_results['F1 Score'][1]),
                'Precision': format_metric_with_std(bootstrap_results['Precision'][0], bootstrap_results['Precision'][1]),
                f'F{beta} Score': f"{fbeta:.3f}"
            }
        except Exception as e:
            print(f"Error evaluating {name} on test set: {str(e)}")
            continue
    
    return pd.DataFrame(results).T

def load_train_test_data(df, time_point):
    """
    Load the patient IDs associated with the train and test sets
    """
    px_ids = {}
    
    for split in ["train", "test"]:
        with open(f"/AIH_recurrence_ML/data/data_splits_{time_point}/{split}_split.txt") as f:
            px_ids[split] = [float(id) for id in f]

    train_df = df[df["rAIHID"].isin(px_ids["train"])]
    test_df = df[df["rAIHID"].isin(px_ids["test"])]
    
    return train_df, test_df

if __name__ == "__main__":
    # Load data
    df_all = pd.read_csv('/AIH_recurrence_ML/data/AIH_processed_all.csv', index_col=0)
    
    # Select relevant columns
    df_all = df_all[['rAIHID', 'rec', 'Gender_0_F_1_M', 'Ethnicity4Groups_0.0',
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
       '1y_binary_tac_trough', '1y_binary_mTOR_trough']]
    
    # Load train and test data
    try:
        train_df, test_df = load_train_test_data(df_all, '1y')
    except FileNotFoundError:
        # If the specific split files aren't found, create a simple 80/20 split
        print("Train/test split files not found, creating random 80/20 split")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['rec'])
    
    # Check class distribution
    print("\nClass distribution in training set:")
    print(train_df['rec'].value_counts())
    print(f"Imbalance ratio: 1:{train_df['rec'].value_counts()[0]/train_df['rec'].value_counts()[1]:.1f}")
    
    print("\nClass distribution in test set:")
    print(test_df['rec'].value_counts())
    print(f"Imbalance ratio: 1:{test_df['rec'].value_counts()[0]/test_df['rec'].value_counts()[1]:.1f}")
    
    # Prepare data
    X_train = train_df.drop(['rAIHID', 'rec'], axis=1)
    y_train = train_df['rec']
    X_test = test_df.drop(['rAIHID', 'rec'], axis=1)
    y_test = test_df['rec']
    
    # Train models with class weights and voting classifiers
    print("\n=== Training models with class weights and voting classifiers ===")
    best_models, metadata, train_results = train_models_with_class_weights_and_voting(
        X_train=X_train,
        y_train=y_train,
        beta=2
    )
    
    # Evaluate on test set with bootstrapping
    print("\n=== Evaluating models on test set with bootstrapping ===")
    test_results = evaluate_models_with_bootstrap(
        models=best_models,
        X_test=X_test,
        y_test=y_test,
        beta=2
    )
    
    # Save models
    save_path = save_best_models(train_results, best_models, dataset='class_weights_voting_ensemble')
    
    # Display results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print("\n=== Cross-Validation Results (Training) ===")
    print(train_results)
    
    print("\n=== Bootstrap Results (Test) ===")
    print(test_results)
    
    # Save metadata and test results
    pd.DataFrame([metadata]).to_csv(f"{save_path}/metadata.csv")
    test_results.to_csv(f"{save_path}/test_bootstrap_results.csv")
    
    print(f"\nModel training and evaluation complete. All files saved to {save_path}")
