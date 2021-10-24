""" Preprocessing Data for SVM and LR Classifier"""

import sys

import pandas as pd
import numpy as np

random_state = 25
frac = 0.2

def preprocess(input_file):
    df = pd.read_csv(str(input_file))

    # use only 6500 rows
    df = df.head(min(6500, df.shape[0]))
    # print(df.shape)

    # 1(i) Removing Quotes, Lowercasing and Normalization
    df[['race', 'race_o', 'field']] = df[['race', 'race_o', 'field']].applymap(lambda x: x.strip("'"))

    df['field'] = df['field'].apply(lambda x: x.lower())

    preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', \
    'funny_important', 'ambition_important', 'shared_interests_important']
    preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', \
    'pref_o_ambitious', 'pref_o_shared_interests']
    preference_scores_of_participant_sum = df[preference_scores_of_participant].sum(axis = 1)
    preference_scores_of_partner_sum = df[preference_scores_of_partner].sum(axis = 1)
    df[preference_scores_of_participant] = df[preference_scores_of_participant].div(preference_scores_of_participant_sum, axis = 0)
    df[preference_scores_of_partner] = df[preference_scores_of_partner].div(preference_scores_of_partner_sum, axis = 0)

    #1(ii) Converting categorical to one-hot vectors
    onehot_encoded = {}
    for attribute in ['gender', 'race', 'race_o', 'field']:
        unique_count = df[attribute].unique()
        unique_count_sorted = np.sort(unique_count)
        # print(unique_count)
        onehot_encoded[attribute] = {}
        for idx, attribute_val in enumerate(unique_count_sorted):
            onehot_encoded[attribute][attribute_val] = [1 if idx == idx_ else 0 for idx_ in range(len(unique_count_sorted)-1)]
    print(f"Mapped vector for female in column gender: {onehot_encoded['gender']['female']}.")
    print(f"Mapped vector for Black/African American in column race: {onehot_encoded['race']['Black/African American']}.")
    print(f"Mapped vector for Other in column race o: {onehot_encoded['race_o']['Other']}.")
    print(f"Mapped vector for economics in column field: {onehot_encoded['field']['economics']}.")

    df_label = df[['decision']]
    df.drop(columns = 'decision', inplace=True)
    # this is esentially converting each attribute column into n-1 columns with values specified as above vector
    for attribute in ['gender', 'race', 'race_o', 'field']:
        df_encoded = pd.get_dummies(df[attribute], prefix=attribute)
        df_encoded_drop_last = df_encoded.drop(columns=df_encoded.columns[-1])
        df = pd.concat([df, df_encoded_drop_last], axis=1)
        df.drop(columns=attribute, axis=1, inplace=True)
    df = pd.concat([df, df_label], axis=1)

    #1(ii) random sampling
    test_df = df.sample(frac = frac, random_state = random_state)
    train_df = df.drop(test_df.index)

    return train_df, test_df

if __name__ == '__main__':
    input_file = 'dating-full.csv'
    train_df, test_df = preprocess(input_file)
    train_df.to_csv('trainingSet.csv', index=False)
    test_df.to_csv('testSet.csv', index=False)
