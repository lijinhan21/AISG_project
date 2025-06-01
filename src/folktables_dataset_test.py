import folktables
from folktables import ACSDataSource, ACSIncome
from sklearn.linear_model import LogisticRegression
import numpy as np

ACSIncomeNew = folktables.BasicProblem(
    features=[
        # 'COW',
        'SCHL',
        # 'MAR',
        'OCCP',
        # 'POBP',
        # 'RELP',
        'WKHP',
        'SEX',
        'AGEP',
        # 'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 10000,    
    group='RAC1P',
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

data_source = ACSDataSource(survey_year='2021', horizon='1-Year', survey='person')
ct_data = data_source.get_data(states=["CT", "PR", "CA", "MI", "NY"], download=True)
pr_data = data_source.get_data(states=["PR"], download=True)
ca_data = data_source.get_data(states=["CA"], download=True)
mi_data = data_source.get_data(states=["MI"], download=True)
tx_data = data_source.get_data(states=["TX"], download=True)

ct_features, ct_labels, ct_env = ACSIncomeNew.df_to_numpy(ct_data)
pr_features, pr_labels, pr_env = ACSIncomeNew.df_to_numpy(pr_data)
ca_features, ca_labels, ca_env = ACSIncomeNew.df_to_numpy(ca_data)
mi_features, mi_labels, mi_env = ACSIncomeNew.df_to_numpy(mi_data)
tx_features, tx_labels, tx_env = ACSIncomeNew.df_to_numpy(tx_data)

print(ct_features.shape, ct_labels.shape, ct_env.shape)
print(ct_features[0], ct_labels[0], ct_env[0])
print(ct_features[1], ct_labels[1], ct_env[1])
print(ct_features[2], ct_labels[2], ct_env[2])
print(ct_features[3], ct_labels[3], ct_env[3])
print(ct_features[4], ct_labels[4], ct_env[4])
print(ct_features[5], ct_labels[5], ct_env[5])
print(ct_features[6], ct_labels[6], ct_env[6])
print(ct_features[7], ct_labels[7], ct_env[7])
print(ct_features[8], ct_labels[8], ct_env[8])

# all together mean and std
all_mean = np.concatenate([ct_features, pr_features, ca_features, mi_features, tx_features], axis=0).mean(axis=0)
all_std = np.concatenate([ct_features, pr_features, ca_features, mi_features, tx_features], axis=0).std(axis=0)
print(all_mean, all_std)

# Plug-in your method for tabular datasets
model = LogisticRegression()

# Train on CT env 1 data
print("Training on CT env 1 data...")
training_feature = ct_features[ct_env == 1]
training_labels = ct_labels[ct_env == 1]
model.fit(training_feature, training_labels)

# Evaluate on CT env 1 and 2 data
print("Model accuracy on CT env 1 data:", model.score(ct_features[ct_env == 1], ct_labels[ct_env == 1]))
print("Model accuracy on CT env 2 data:", model.score(ct_features[ct_env == 2], ct_labels[ct_env == 2]))


# Train on CT data
print("Training on CT data...")
training_feature = ct_features
training_labels = ct_labels
model.fit(training_feature, training_labels)

# Evaluate on CT env 1 and 2 data
print("Model accuracy on CT env 1 data:", model.score(ct_features[ct_env == 1], ct_labels[ct_env == 1]))
print("Model accuracy on CT env 2 data:", model.score(ct_features[ct_env == 2], ct_labels[ct_env == 2]))
print("Model accuracy on PR env 1 data:", model.score(pr_features[pr_env == 1], pr_labels[pr_env == 1]))
print("Model accuracy on PR env 2 data:", model.score(pr_features[pr_env == 2], pr_labels[pr_env == 2]))
print("Model accuracy on CA env 1 data:", model.score(ca_features[ca_env == 1], ca_labels[ca_env == 1]))
print("Model accuracy on CA env 2 data:", model.score(ca_features[ca_env == 2], ca_labels[ca_env == 2]))
print("Model accuracy on MI env 1 data:", model.score(mi_features[mi_env == 1], mi_labels[mi_env == 1]))
print("Model accuracy on MI env 2 data:", model.score(mi_features[mi_env == 2], mi_labels[mi_env == 2]))
print("Model accuracy on TX env 1 data:", model.score(tx_features[tx_env == 1], tx_labels[tx_env == 1]))

exit(0)

# Evaluate on CT data
print("Model accuracy on CT data:", model.score(ct_features, ct_labels))

# Test on PR, CA, MI, TX data
print(f"Model accuracy on PR data: {model.score(pr_features, pr_labels)}")
print(f"Model accuracy on CA data: {model.score(ca_features, ca_labels)}")
print(f"Model accuracy on MI data: {model.score(mi_features, mi_labels)}")
print(f"Model accuracy on TX data: {model.score(tx_features, tx_labels)}")