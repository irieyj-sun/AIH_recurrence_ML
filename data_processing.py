import pandas as pd
import numpy as np
from datetime import datetime
from tableone import TableOne
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('Recurrent_AIH.csv', encoding='latin-1')
print(df.shape)

# set upper limit of tacrolimus trough level to 30 
df['Tacrolimus_initially_post_Tx_trough_levels_ng_mL'] = np.clip(df['Tacrolimus_initially_post_Tx_trough_levels_ng_mL'], None, 30)
print(df.shape)

df_recurrence = df[(df['LiverTxRecurrencemonths'] >= 12) | pd.isna(df_study1['LiverTxRecurrencemonths'])]
#create MMF_AZA combined variable
df_recurrence['MMF_AZA']= np.logical_or(df_recurrence['MMF_initially_post_Tx'],df_recurrence['AZA_initially_post_Tx']).astype(int)
print(df_recurrence .shape)
print(df_recurrence['rec'].value_counts()/len(df_recurrence))

def initial_imm_regimen(row):
    # single imm: tac or cyclo only 
    criterion1 = (row['Tacrolimus_initially_post_Tx'] == 1) or (row['Cyclosporine_initially_post_Tx'] == 1)
    # double imm
    criterion2 = criterion1 and (row['MMF_initially_post_Tx'] == 1)
    # triple imm 
    criterion3 = criterion2 and (row['PDN_initially_post_Tx'] == 1)

    if criterion3:
        return 3 
    elif criterion2:
        return 2 
    elif criterion1:
        return 1  
    else:
        return 0  

df_recurrence['imm_regimen_initial'] = df_recurrence.apply(initial_imm_regimen, axis=1)

def oneyear_imm_regimen(row):
    # single imm: tac or cyclo only 
    criterion1 = (row['Tacrolimus_1_y_post_Tx'] == 1) or (row['Cyclosporine_1_y_post_Tx'] == 1)
    # double imm
    criterion2 = criterion1 and (row['MMF_1_y_post_Tx'] == 1)
    # triple imm 
    criterion3 = criterion2 and (row['PDN_1_y_post_Tx'] == 1)

    if criterion3:
        return 3 
    elif criterion2:
        return 2 
    elif criterion1:
        return 1  
    else:
        return 0  

df_recurrence['imm_regimen_1y'] = df_recurrence.apply(oneyear_imm_regimen, axis=1)

class FeatureExtractor():
    
    def extract_study1_features(self,df):
        """ 
        Accepts a dataframe as input, extract columns for study aim 1 
        and format them appropriately so that a learning model
        may be trained on them.
        """

        #outcome variable: graft loss?/recurrence in AIH? check distribution 
        out = pd.DataFrame(df['rAIHID'])

        out = pd.concat([out, df['rec'].fillna(0)], axis=1)
        
        # baseline 
        out = pd.concat([out, df['Gender_0_F_1_M']], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(df['Ethnicity4Groups'], prefix='Ethnicity4Groups',dtype=int)], axis=1)
        out = pd.concat([out, df['Ageatdiagnosis'].fillna(df['Ageatdiagnosis'].mean())], axis=1)
        out = pd.concat([out, df['Ageattransplantation'].fillna(df['Ageattransplantation'].mean())], axis=1)
        out = pd.concat([out, df['AIH_type'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['Overlap']], axis=1)
        out = pd.concat([out, df['Concomitant_autoimmune_diseases'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['ANA_pre_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['ASMA_pre_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['AMA_pre_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(df['ABO_recipient'], prefix='ABO_recipient',dtype=int)], axis=1)
        out = pd.concat([out, df['Budesonide_pre_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['Esophageal_varices_pre_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['Variceal_hemorrhage_pre_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['Hepatic_encephalopathy_pre_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['Ascites_pre_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['TxmonthsafterDx'].fillna(df['TxmonthsafterDx'].mean())], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(df['Tx_type'], prefix='Tx_type',dtype=int)], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(df['LT_Indication3Gropus'], prefix='LT_Indication3Gropus',dtype=int)], axis=1)
        out = pd.concat([out, 
            pd.get_dummies(df['Anastomosis_type'], prefix='Anastomosis_type',dtype=int)], axis=1)
        out = pd.concat([out, df['Explant_Bx_fibrosis'].apply(lambda x: 1 if x in [3, 4] else 0)],axis=1)
        out = pd.concat([out, df['Explant_Bx_plasma_cells'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['Explant_Bx_necroinflammatory_activity'].apply(lambda x: 1 if x in [3, 4] else 0)],axis=1)
        
        # donor and mathch
        out = pd.concat([out, df['Donor_age'].fillna(df['Donor_age'].mean())], axis=1)
        out = pd.concat([out, df['Donor_gender'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['GenderMismatch'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['IgG_pre_Tx_ULN_A'].fillna(df['IgG_pre_Tx_ULN_A'].median())], axis=1)
        out = pd.concat([out, df['IgA_pre_Tx_ULN'].fillna(df['IgA_pre_Tx_ULN'].median())], axis=1)
        out = pd.concat([out, df['IgM_pre_Tx_ULN'].fillna(df['IgM_pre_Tx_ULN'].median())], axis=1)
        
        # at transplant and static post-transplant
        out = pd.concat([out, df['Rejection'].fillna(0.5)], axis=1) #rejection at all, acute or chronic 
        #out = pd.concat([out, df['RejectionAcute'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['Sepsis_post_Tx'].fillna(0.5)], axis=1)
        # out = pd.concat([out, df['Bone_fractures_post_Tx'].fillna(0.5)], axis=1)


        #longitudinal follow-up
        out = pd.concat([out, df['ALT_pre_Tx'].fillna(df['ALT_pre_Tx'].mean())], axis=1)
        out = pd.concat([out, df['AST_pre_Tx'].fillna(df['AST_pre_Tx'].mean())], axis=1)
        out = pd.concat([out, df['ALP_pre_Tx'].fillna(df['ALP_pre_Tx'].mean())], axis=1)
        out = pd.concat([out, df['Bilirubin_pre_Tx'].fillna(df['Bilirubin_pre_Tx'].mean())], axis=1)
        out = pd.concat([out, df['INR_pre_Tx'].fillna(df['INR_pre_Tx'].mean())], axis=1)
        out = pd.concat([out, df['Creatitine_pre_Tx'].fillna(df['Creatitine_pre_Tx'].mean())], axis=1)
        out = pd.concat([out, df['MELD_pre_Tx'].fillna(df['MELD_pre_Tx'].mean())], axis=1)
        
        out = pd.concat([out, df['ALT_3_m'].fillna(df['ALT_3_m'].mean())], axis=1)
        out = pd.concat([out, df['AST_3'].fillna(df['AST_3'].mean())], axis=1)
        out = pd.concat([out, df['ALP_3_m'].fillna(df['ALP_3_m'].mean())], axis=1)
        out = pd.concat([out, df['Bilirubin_3_m'].fillna(df['Bilirubin_3_m'].mean())], axis=1)
        out = pd.concat([out, df['ALT_6_m'].fillna(df['ALT_6_m'].mean())], axis=1)
        out = pd.concat([out, df['AST_6_m'].fillna(df['AST_6_m'].mean())], axis=1)
        out = pd.concat([out, df['ALP_6_m'].fillna(df['ALP_6_m'].mean())], axis=1)
        out = pd.concat([out, df['Bilirubin_6_m'].fillna(df['Bilirubin_6_m'].mean())], axis=1)
        out = pd.concat([out, df['ALT_1_y'].fillna(df['ALT_1_y'].mean())], axis=1)
        out = pd.concat([out, df['AST_1_y'].fillna(df['AST_1_y'].mean())], axis=1)
        out = pd.concat([out, df['ALP_1_y'].fillna(df['ALP_1_y'].mean())], axis=1)
        out = pd.concat([out, df['Bilirubin_1_y'].fillna(df['Bilirubin_1_y'].mean())], axis=1)


        out = pd.concat([out, df['Prednisone_pre_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['AZA_pre_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['MMF_pre_Tx'].fillna(0.5)], axis=1)

        out = pd.concat([out, df['Tacrolimus_initially_post_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['Cyclosporine_initially_post_Tx'].fillna(0.5)], axis=1)
        # out = pd.concat([out, df['CalcineurinInhibitor'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['mTOR_inhibitors_initially_post_Tx'].fillna(0.5)], axis=1)
        # out = pd.concat([out, df['MMF_initially_post_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['MMF_AZA'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['PDN_initially_post_Tx'].fillna(0.5)], axis=1)
        out = pd.concat([out, df['PDN_1_y_post_Tx'].fillna(0.5)], axis=1)

        out = pd.concat([out, df['Cyclosporine_initially_post_Tx_trough_levels_ng_mL']], axis=1)
        out = pd.concat([out, df['Tacrolimus_initially_post_Tx_trough_levels_ng_mL']], axis=1)
        out = pd.concat([out, df['mTOR_inhibitors_initially_post_Tx_trough_levels_ng_mL']], axis=1)
        out = pd.concat([out, df['Cyclosporine_1_y_post_Tx_trough_levels_ng_mL']], axis=1)
        out = pd.concat([out, df['Tacrolimus_1_y_post_Tx_trough_levels_ng_mL']], axis=1)
        out = pd.concat([out, df['mTOR_inhibitors_1_y_post_Tx_trough_levels_ng_mL']], axis=1)

        out = pd.concat([out, df['imm_regimen_initial']], axis=1)
        out = pd.concat([out, df['imm_regimen_1y']], axis=1)
       
        return out 

extractor = FeatureExtractor()
processed_data_1y = extractor.extract_study1_features(df_recurrence)

processed_data_1y['initial_binary_cyc_trough']= processed_data_1y['Cyclosporine_initially_post_Tx_trough_levels_ng_mL'].apply(lambda x: 1 if x > 224.0 else 0)
processed_data_1y['initial_binary_tac_trough']= processed_data_1y['Tacrolimus_initially_post_Tx_trough_levels_ng_mL'].apply(lambda x: 1 if x > 10.0 else 0)
processed_data_1y['initial_binary_mTOR_trough']= processed_data_1y['mTOR_inhibitors_initially_post_Tx_trough_levels_ng_mL'].apply(lambda x: 1 if x > 4.5 else 0)
processed_data_1y['1y_binary_cyc_trough']= processed_data_1y['Cyclosporine_1_y_post_Tx_trough_levels_ng_mL'].apply(lambda x: 1 if x > 161.0 else 0)
processed_data_1y['1y_binary_tac_trough']= processed_data_1y['Tacrolimus_1_y_post_Tx_trough_levels_ng_mL'].apply(lambda x: 1 if x > 6.0 else 0)
processed_data_1y['1y_binary_mTOR_trough']= processed_data_1y['mTOR_inhibitors_1_y_post_Tx_trough_levels_ng_mL'].apply(lambda x: 1 if x > 6.0 else 0)


processed_data_1y = processed_data_1y.drop(['Cyclosporine_initially_post_Tx_trough_levels_ng_mL',
                                            'Tacrolimus_initially_post_Tx_trough_levels_ng_mL',
                                            'mTOR_inhibitors_initially_post_Tx_trough_levels_ng_mL',
                                            'Cyclosporine_1_y_post_Tx_trough_levels_ng_mL',
                                            'Tacrolimus_1_y_post_Tx_trough_levels_ng_mL',
                                            'mTOR_inhibitors_1_y_post_Tx_trough_levels_ng_mL'],axis=1)

processed_data_1y['delta_ALT_tx_to_3m'] = processed_data_1y['ALT_3_m']-processed_data_1y['ALT_pre_Tx']
processed_data_1y['delta_AST_tx_to_3m'] = processed_data_1y['AST_3']-processed_data_1y['AST_pre_Tx']
processed_data_1y['delta_ALP_tx_to_3m'] = processed_data_1y['ALP_3_m']-processed_data_1y['ALP_pre_Tx']
processed_data_1y['delta_bili_tx_to_3m'] = processed_data_1y['Bilirubin_3_m']-processed_data_1y['Bilirubin_pre_Tx']

processed_data_1y['delta_ALT_3m_to_6m'] = processed_data_1y['ALT_6_m']-processed_data_1y['ALT_3_m']
processed_data_1y['delta_AST_3m_to_6m'] = processed_data_1y['AST_6_m']-processed_data_1y['AST_3']
processed_data_1y['delta_ALP_3m_to_6m'] = processed_data_1y['ALP_6_m']-processed_data_1y['ALP_3_m']
processed_data_1y['delta_bili_3m_to_6m'] = processed_data_1y['Bilirubin_6_m']-processed_data_1y['Bilirubin_3_m']

processed_data_1y['delta_ALT_6m_to_1y'] = processed_data_1y['ALT_1_y']-processed_data_1y['ALT_6_m']
processed_data_1y['delta_AST_6m_to_1y'] = processed_data_1y['AST_1_y']-processed_data_1y['AST_6_m']
processed_data_1y['delta_ALP_6m_to_1y'] = processed_data_1y['ALP_1_y']-processed_data_1y['ALP_6_m']
processed_data_1y['delta_bili_6m_to_1y'] = processed_data_1y['Bilirubin_1_y']-processed_data_1y['Bilirubin_6_m']

#percentage cahnge from 6 months to 1 year 
processed_data_1y['percent_delta_ALT_6m_to_1y'] = (processed_data_1y['delta_ALT_6m_to_1y']/processed_data_1y['ALT_6_m'])*100
processed_data_1y['percent_delta_AST_6m_to_1y'] = (processed_data_1y['delta_AST_6m_to_1y']/processed_data_1y['AST_6_m'])*100
processed_data_1y['percent_delta_ALP_6m_to_1y'] = (processed_data_1y['delta_ALP_6m_to_1y']/processed_data_1y['ALP_6_m'])*100
processed_data_1y['percent_delta_bili_6m_to_1y'] = (processed_data_1y['delta_bili_6m_to_1y']/processed_data_1y['Bilirubin_6_m'])*100

processed_data_1y.to_csv('/AIH_recurrence_ML/data/AIH_processed_all.csv')

train_splt = 0.7

np.random.seed(1234)

patient_identifiers = np.array(processed_data_1y_adults["rAIHID"])

np.random.shuffle(patient_identifiers)

# Split into train and test
train, test = np.split(patient_identifiers, [int(train_splt * len(patient_identifiers))])

# Save the splits to text files
with open("/AIH_recurrence_ML/data/data_splits_adults/train_split.txt", "w") as f:
    f.write("\n".join(train.astype('str')))

with open("/AIH_recurrence_ML/data/data_splits_adults/test_split.txt", "w") as f:
    f.write("\n".join(test.astype('str')))
