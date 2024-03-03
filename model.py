import pandas as pd, numpy as np, plotly.express as px
import plotly.graph_objects as go  
from plotly.subplots import make_subplots 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, recall_score 
from sklearn.preprocessing import FunctionTransformer
import shap 
import pickle

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

train = pd.read_csv('train.csv') 
test = pd.read_csv('test.csv')

def inspect(df):
    print('\n')
    print('Missing Values: ')
    print(df.isnull().sum())
    print('\n')
    print('Duplicated Values: ')    
    print(df.duplicated().sum())
    print('\n')
    print('Data Types: ')
    print(df.dtypes)
    print('\n')
    print(f'Rows: {df.shape[0]}')   
    print(f'Attributes: {df.shape[1]}')
    print('\n')
    print('Head: ')
    print('\n')
    return df.head()

inspect(train)
inspect(test)

train_duplicated = train[train.duplicated(keep=False)]
train_duplicated

test_duplicated = test[test.duplicated(keep=False)]
test_duplicated 

train.drop_duplicates(inplace = True)
test.drop_duplicates(inplace = True)

print('\n')
print(f'Train new rows count: {train.shape[0]}')
print(f'Test new rows count: {test.shape[0]}')

def categorize_features(df):
    
    continuous_features = []
    binary_features = []
    
    for col in df.columns:
        if df[col].nunique() <= 2: 
            binary_features.append(col)
        else:
            continuous_features.append(col) 
    return continuous_features, binary_features

continuous_features, binary_features = categorize_features(train)

print('\n')
print('Continuous features:')
print(continuous_features)
print('\n')
print('Binary features:')
print(binary_features)

binary_features.remove('fake') 

legend_df = train.copy()
legend_df['fake'] = legend_df['fake'].replace({0: 'Real Accounts', 1: "Fake Accounts"}) 


# def create_barplots(df, legend_df):
#     for feature in binary_features:
#         fig = px.histogram(train, x=feature, color=legend_df['fake'],
#                            color_discrete_sequence=['#636EFA','#EF553B'],
#                            barmode='group', template='plotly_white',labels={'color': 'Real/Fake'})
        
#         fig.update_layout(title=f'{feature}?',
#                           xaxis_title=feature, yaxis_title='Count', xaxis=dict(tickmode='array',
#                                                                                tickvals=[0, 1],
#                                                                                ticktext=['No', 'Yes']
#                                                                                ),
#                      height = 650)
        
#         fig.show()

# create_barplots(train,legend_df)

# fig, axes = plt.subplots(nrows=len(continuous_features), ncols=2, figsize=(10, 30))

# for i, col in enumerate(continuous_features):
#     sns.boxplot(data=train[train['fake'] == 0], x=col, ax=axes[i, 0], color = '#7393B3')
#     sns.boxplot(data=train[train['fake'] == 1], x=col, ax=axes[i, 1], color = 'orange')
#     axes[i, 0].set_title(col + " - Real Accounts")
#     axes[i, 1].set_title(col + " - Fake Accounts")

# plt.tight_layout()
# plt.show()

# fig = px.pie(legend_df, names='fake', title='Target variable distribution', color_discrete_sequence = ['#636EFA','#EF553B'])
# fig.update_layout(template = 'ggplot2')
# fig.show()

# Training, predicting and evaluating baseline

# Splitting dataset into independent variables (X) and target variable (y)
X = train.drop('fake', axis = 1)
y = train['fake']

# Creating training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Initializing mode
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train) # Fitting to training data 

y_pred = rf.predict(X_val) # Predicting on validation set
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})

baseline_score = roc_auc_score(y_val, y_pred)
print('\n')
print('AUC-ROC Baseline: ', baseline_score.round(2))
print('\n')

sns.set_style('darkgrid')
sns.lineplot(x='FPR', y='TPR', data=roc_df, label=f'RandomForest Classifier(AUC-ROC = {baseline_score.round(2)})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing')
plt.title('AUC-ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# plt.show()
print('\n')
print('y_val value counts')
print(y_val.value_counts())
print('\n')
print('predicted value counts')
print(np.unique(y_pred, return_counts=True))

shap_values = shap.TreeExplainer(rf).shap_values(X_val)
# shap.summary_plot(shap_values, X_val, plot_type="bar")

train['activity ratio'] = np.round(train['#posts'] / train['#followers'], 2)

# Does the account have more followers than follows?
train['#followers > #follows?'] = (train['#followers'] > train['#follows']).astype(int)

# print(train)

fig = px.histogram(train, x=train['#followers > #follows?'], color=legend_df['fake'],
                   color_discrete_sequence=['#636EFA','#EF553B'],
                   barmode='group', template='plotly_white',labels={'color': 'Real/Fake'})
        
fig.update_layout(title='More Followers than Follows Distribution',
                  xaxis_title='#followers > #follows?', yaxis_title='Count', xaxis=dict(tickmode='array',
                  tickvals=[0, 1],
                  ticktext=['No', 'Yes']),
                  height = 800)
        
# fig.show()

fig = px.box(train, x='fake', y='activity ratio', color = legend_df['fake'], title = "Activity Ratio")

fig.update_layout(xaxis_title="Real/Fake", yaxis_title='Count', xaxis=dict(tickmode='array',
                                                                               tickvals=[0, 1],
                                                                               ticktext=['Real Accounts', 'Fake Accounts']
                                                                               ),
                     height = 650)


# fig.show()

train.isnull().sum()

train.isin([np.inf, -np.inf]).sum()

train.replace([np.inf, -np.inf], np.nan, inplace=True)

train.dropna(inplace=True)

df_means = train.mean().round(2)
df_stds = train.std().round(2)
results = pd.concat([df_means, df_stds], axis = 1)
results.columns = ['Mean', 'Standard Deviation']

# print(results)

X = train.drop('fake', axis = 1)
y = train.fake

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                 test_size = 0.35, random_state = 123)

models = {
    "XGBoost": XGBClassifier(random_state = 42),
    "LGBM": LGBMClassifier(random_state = 42),
    "CatBoost": CatBoostClassifier(verbose=False, random_state = 42),
    "AdaBoost": AdaBoostClassifier(random_state = 42)
}

pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline([
        ("scaler", StandardScaler()), # Rescaling data
        ("model", model) # Initializing model
    ])

results = {}
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    results[name] = {
        "pipeline": pipeline,
        "auc": auc
    }
    print(f"{name}: AUC-ROC score = {auc:.2f}")

plt.figure(figsize=(8, 6))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_val, result["pipeline"].predict(X_val))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label = 'Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# plt.show()

test['activity ratio'] = np.round(test['#posts'] / test['#followers'], 2)

test['#followers > #follows?'] = (test['#followers'] > test['#follows']).astype(int)

test.replace([np.inf, -np.inf], np.nan, inplace=True)

test.dropna(inplace=True)
# print(test)

X = test.drop('fake', axis = 1) 
y = test.fake

print(X)

y.value_counts()

catboost_pipeline = results["CatBoost"]["pipeline"] 

y_pred = catboost_pipeline.predict(X)

auc = roc_auc_score(y, y_pred)
print(f"CatBoos: AUC-ROC score on unseen data = {auc:.4f}")

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y, catboost_pipeline.predict(X))
plt.plot(fpr, tpr, label=f"(CatBoost AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label = 'Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# plt.show()

print("Value counts for y:")
print(y.value_counts())
print('\n')
print("Value counts for y_pred:")
print(pd.Series(y_pred).value_counts())
print('\n')

cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
# plt.show()

print('\n')
print(f'Recall Score: {np.round(recall_score(y, y_pred),2) * 100}%')

model = catboost_pipeline.named_steps['model']
explainer = shap.Explainer(model, X_train)

shap_values = explainer(X)

shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title('CatBoost - Feature Importance')
plt.tight_layout()
# plt.show()



with open('catboost_model.pkl', 'wb') as f:
    pickle.dump(catboost_pipeline, f)





