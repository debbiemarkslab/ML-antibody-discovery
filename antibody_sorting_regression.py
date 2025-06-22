#!/usr/bin/env python
# coding: utf-8

# In this notebook, we train the logistic regression models on the affinity sorting data (FACS1 vs. MACS1), score all the sequences, and select the top FACS1 sequences according to the model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from Levenshtein import distance

import sklearn.linear_model as lm
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
import sklearn
print("sklearn version:", sklearn.__version__)

# In[1]:


IPI_VH_SEQS = ['VH1-69']
IPI_VL_SEQS = ['VK1-39', 'VK3-15', 'VK3-20', 'VK4-01']
IPI_3_VL_SEQS = ['VK1-39', 'VK3-15', 'VK3-20']

sorts = [
    {
        "filename": "zenodo_data/hROBO1_Test1_Lib31.csv",
        "SPR_antigens": ["hROBO1"],
        "SPR_antigens2": ["ROBO1"],
        "light": IPI_3_VL_SEQS,
        "heavy": IPI_VH_SEQS,
    },
    {
        "filename": "zenodo_data/hROBO2N_Test1_Lib31.csv",
        "SPR_antigens": ["hROBO2N"],
        "SPR_antigens2": ["ROBO2N"],
        "light": IPI_3_VL_SEQS,
        "heavy": IPI_VH_SEQS,
    },
    {
        "filename": "zenodo_data/PD1-L1_Test1_Lib31.csv",
        "SPR_antigens": ["PD1-L1", "PD1-L1_Test1_Lib31"],
        "SPR_antigens2": ["PDL1"],
        "light": IPI_3_VL_SEQS,
        "heavy": IPI_VH_SEQS,
    },
    {
        "filename": "zenodo_data/hPD-L2_chase_Test1_Lib31.csv",
        "SPR_antigens": ["hPD-L2"],
        "SPR_antigens2": ["PDL2"],
        "light": IPI_3_VL_SEQS,
        "heavy": IPI_VH_SEQS,
    }
]


# In[5]:


alphabet = "ACDEFGHIKLMNPQRSTVWY-"

def add_cdr3_gaps(s, max_len=20):
    s2 = s[1:]
    cut_len_L = min((len(s2)+1) // 2, max_len // 2)
    cut_len_R = min(len(s2) // 2, (max_len-1) // 2)
    gap_len = max_len - 1 - cut_len_L - cut_len_R
    s2 = s[0] + s2[:cut_len_L] + "-" * gap_len + s2[-cut_len_R:]
    return s2

def vectorize_cdr3(s, max_len=20):
    s = add_cdr3_gaps(s, max_len=max_len)
    v = np.zeros((len(s), len(alphabet)), dtype=np.int8)
    for i, c in enumerate(s):
        v[i, alphabet.index(c)] = 1
    return v

assert len(add_cdr3_gaps("CARAPQYGLGYSAYFDI")) == 20
assert add_cdr3_gaps("CARAPQYGLGYSAYFDI").replace("-", "") == "CARAPQYGLGYSAYFDI"

def cdr3_seqs_to_onehot(seqs, max_len=20):
    onehot_arr = np.zeros((len(seqs), max_len * len(alphabet)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        onehot_arr[i] = vectorize_cdr3(seq).flatten()

    labels = [f"{i}{a}" for i in range(1, max_len+1) for a in alphabet]
    return onehot_arr, labels

def filter_cdr3(seqs: pd.Series):
#     return s[:3] == "CAR" and s[-3:] in {"FDY", "LDY", "FDI", "FDP"}
    return (seqs.str[:3] == "CAR") & seqs.str[-3:].isin({"FDY", "LDY", "FDI", "FDP"})


# In[6]:


def get_kmer_list(seq, include_framework=''):
    if 'C' in include_framework:
        seq = 'C' + seq
    if 'W' in include_framework:
        seq = seq + 'W'
    kmer_counts = {}

    kmer_len = 1
    num_chunks = (len(seq)-kmer_len)+1
    for idx in range(0,num_chunks):
        kmer = seq[idx:idx+kmer_len]
        assert len(kmer) == kmer_len
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1

    kmer_len = 2
    num_chunks = (len(seq)-kmer_len)+1
    for idx in range(0,num_chunks):
        kmer = seq[idx:idx+kmer_len]
        assert len(kmer) == kmer_len
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1

    kmer_len = 3
    num_chunks = (len(seq)-kmer_len)+1
    for idx in range(0,num_chunks):
        kmer = seq[idx:idx+kmer_len]
        assert len(kmer) == kmer_len
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1
    #print kmer_counts
    return [(key,val) for key,val in kmer_counts.items()]


# In[7]:


cdr3_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
kmer_to_idx = {}
counter = 0
kmer_list = [aa for aa in cdr3_alphabet]
for aa in cdr3_alphabet:
    for bb in cdr3_alphabet:
        kmer_list.append(aa+bb)
        for cc in cdr3_alphabet:
            kmer_list.append(aa+bb+cc)

kmer_to_idx = {aa: i for i, aa in enumerate(kmer_list)}


# In[8]:


def cdr3_seqs_to_arr(seqs, include_framework=''):
    seq_to_kmer_vector = {}
    for seq in seqs:
        # Make into kmers
        kmer_data_list = get_kmer_list(seq, include_framework=include_framework)
        norm_val = 0.
        for kmer,count in kmer_data_list:
            count = float(count)
            norm_val += (count * count)
        norm_val = np.sqrt(norm_val)

        # L2 normalize
        final_kmer_data_list = []
        for kmer,count in kmer_data_list:
            final_kmer_data_list.append((kmer_to_idx[kmer],float(count)/norm_val))

        # save to a dictionary
        seq_to_kmer_vector[seq] = final_kmer_data_list

    kmer_arr = np.zeros((len(seqs), len(kmer_to_idx)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        kmer_vector = seq_to_kmer_vector[seq]
        for j_kmer,val in kmer_vector:
            kmer_arr[i, j_kmer] = val
    return kmer_arr


# In[9]:


def normalize_abundance(df, col):
    s = df[col].fillna(0)
    s = (s / s.sum()) * 1e6
    s[s < 1] = 1
    return s

# def filter_rounds(df, cols):
#     return df[(df[cols] > 5).any(axis=1)]

def calc_enrichment(df, col1, col2, col1_min=None, col2_min=None):
    s1, s2 = normalize_abundance(df, col1), normalize_abundance(df, col2)
    enrichment = np.log(s2) - np.log(s1)
    if col1_min is not None and col2_min is not None:
        enrichment[(df[col1].fillna(0) < col1_min) & (df[col2].fillna(0) < col2_min)] = 0
    elif col1_min is not None:
        enrichment[df[col1].fillna(0) < col1_min] = 0
    elif col2_min is not None:
        enrichment[df[col2].fillna(0) < col2_min] = 0
#     enrichment[(df[[col1, col2]].fillna(0) < 2).all(axis=1)] = 0
    return enrichment


# In[10]:


print("a1/m1", "a2/m1", "a3/m1", "a2/a1", "a3/a1", "file", sep='\t')
for fname in [
    "zenodo_data/hROBO1_Test1_Lib31.csv",
    "zenodo_data/hROBO2N_Test1_Lib31.csv",
    "zenodo_data/PD1-L1_Test1_Lib31.csv",
    "zenodo_data/hPD-L2_chase_Test1_Lib31.csv",
]:
    df = pd.read_csv(fname)
    df = df[filter_cdr3(df["CDR3"])]

    df["Aff1_Macs1"] = calc_enrichment(df, "Macs1", "Aff1")
    df["Aff2_Macs1"] = calc_enrichment(df, "Macs1", "Aff2")
    df["Aff2_Aff1"] = calc_enrichment(df, "Aff1", "Aff2")
    df["Aff3_Macs1"] = calc_enrichment(df, "Macs1", "Aff2")
    df["Aff3_Aff1"] = calc_enrichment(df, "Aff1", "Aff3")
    df["Aff3_Aff2"] = calc_enrichment(df, "Aff2", "Aff3")

    print((df["Aff1_Macs1"] != 0).sum(), (df["Aff2_Macs1"] != 0).sum(), (df["Aff3_Macs1"] != 0).sum(), (df["Aff3_Aff1"] != 0).sum(), (df["Aff3_Aff1"] != 0).sum(), fname, sep='\t')


# # Logistic regression

# ### use kmer representations

# For each sort, train an LR model using CDR3 k-mers + VL gene identities, then score all the antibody sequences in that sort using the model.

# In[ ]:


results = []

for sort in sorts:
    print()
    print(sort)
    fname = sort["filename"]
    target = fname.rsplit("/", 1)[1].split("_", 1)[0]
    light_chains = sort["light"]
    heavy_chains = sort["heavy"]
    antigens = sort["SPR_antigens"]
    antigens2 = sort["SPR_antigens2"]
    params_file = f"params/{fname.rsplit('/', 1)[1].replace('.csv', '')}_kmer_LR.pkl"
    aff3_file = f"scores/{fname.rsplit('/', 1)[1].replace('.csv', '')}_kmer_aff3_scores.csv"

    df = pd.read_csv(fname)
    df = df[filter_cdr3(df["CDR3"])]
    df = df[df["light"].isin(light_chains)]
    df = df[df["heavy"].isin(heavy_chains)]

    if "Macs1" not in df.columns:
        df["Macs1"] = df["Macs1_A1"]
        df["Aff1"] = df["Aff1_A1"]
        df["Aff2"] = df["Aff2_A1"]
        df["Aff3"] = df["Aff3_Combined"]

    df["Aff1_Macs1"] = calc_enrichment(df, "Macs1", "Aff1")
    df["Aff2_Macs1"] = calc_enrichment(df, "Macs1", "Aff2")
    df["Aff2_Aff1"] = calc_enrichment(df, "Aff1", "Aff2")
    df["Aff3_Macs1"] = calc_enrichment(df, "Macs1", "Aff2")
    df["Aff3_Aff1"] = calc_enrichment(df, "Aff1", "Aff3")
    df["Aff3_Aff2"] = calc_enrichment(df, "Aff2", "Aff3")

    char_df = pd.read_csv("zenodo_data/00Library_Biophysics_486Abs(1).csv", header=0, skiprows=1, names="Antibody_Name	Antigen	HC	LC	CDRH3	FACS1 count	FACS2 count	FACS3 count	PSR	SEC	SPR KD	SPR ka	SPR kdis (1/s)	Cell Display EC50".split('\t'))
    char_df["key"] = char_df["CDRH3"] + ":" + char_df["HC"] + ":" + char_df["LC"]
    char_df = char_df[char_df["Antigen"].isin(antigens2)]
    char_df["SPR KD"] = char_df["SPR KD"].replace("Fail", 1000).astype(float)
    char_df["Cell Display EC50"] = char_df["Cell Display EC50"].replace("Fail", 1000).astype(float)

    df = pd.merge(df, char_df, on="key", how="left")

    print("a1/m1", "a2/m1", "a3/m1", "a2/a1", "order", "file", sep='\t')
    print((df["Aff1_Macs1"] != 0).sum(), (df["Aff2_Macs1"] != 0).sum(), (df["Aff3_Macs1"] != 0).sum(), (df["Aff3_Aff1"] != 0).sum(), len(char_df), fname, sep='\t')
    #     assert len(df[(df["Aff1_Macs1"] != 0) & ((df["Macs1"] < 5) | df["Macs1"].isnull()) & ((df["Aff1"] < 5) | df["Aff1"].isnull())]) == 0

    kmer_arr = cdr3_seqs_to_arr(df['CDR3'], include_framework='W')
    # vh_onehot = pd.get_dummies(pd.Categorical(df['heavy'], categories=IPI_VH_SEQS_V2, ordered=True))
    vl_onehot = pd.get_dummies(pd.Categorical(df['light'], categories=IPI_VL_SEQS, ordered=True))
    length_onehot = pd.get_dummies(pd.Categorical(df['CDR3'].str.len(), ordered=True))
    print("loaded kmer arr length:", len(df))
    kmer_vh_vl_arr = np.concatenate([
        kmer_arr,
    #     vh_onehot.values,
        vl_onehot.values
    ], axis=1)
    kmer_vh_vl_len_arr = np.concatenate([kmer_vh_vl_arr, length_onehot.values], axis=1)
    kmer_arr_labels = kmer_list
    kmer_vh_vl_arr_labels = (
        kmer_arr_labels +
    #     vh_onehot.columns.tolist() +
        vl_onehot.columns.tolist()
    )
    kmer_vh_vl_len_arr_labels = kmer_vh_vl_arr_labels + length_onehot.columns.tolist()

    char_kmer_arr = cdr3_seqs_to_arr(char_df['CDRH3'], include_framework='W')
    char_vl_onehot = pd.get_dummies(pd.Categorical(char_df['LC'], categories=IPI_VL_SEQS, ordered=True))
    char_kmer_vh_vl_arr = np.concatenate([
        char_kmer_arr,
        char_vl_onehot.values
    ], axis=1)

    X, X_labels, y = kmer_vh_vl_arr, kmer_vh_vl_arr_labels, df["Aff1_Macs1"].values
    X = pd.DataFrame(X, columns=X_labels)
    X, y = X[y != 0], y[y != 0]
    np.random.seed(1)
    msk = np.random.permutation(len(y)) < int(len(y) * 0.8)
    X_train, X_test = X[msk], X[~msk]
    y_train, y_test = y[msk], y[~msk]
    print(len(y_train), (y_train > 0).sum(), "Train")
    print(len(y_test), (y_test > 0).sum(), "Test")

    X_val, y_val = kmer_vh_vl_arr, df["Aff3_Macs1"].values
    X_val = pd.DataFrame(X_val, columns=X_labels)
    X_val, y_val = X_val[y_val != 0], y_val[y_val != 0]
    print(len(y_val), (y_val > 0).sum(), "Aff3 Val")

    X_val2, y_val2 = char_kmer_vh_vl_arr, ((char_df["SPR KD"] < 100) | (char_df["Cell Display EC50"] < 100))
    X_val2 = pd.DataFrame(X_val2, columns=X_labels)
    print(len(y_val2), (y_val2 > 0).sum(), "SPR Val")

    # train model
    thresh = 0.
    if os.path.exists(params_file):
        with open(params_file, "rb") as f:
            clf = pickle.load(f)
    else:
        clf = lm.LogisticRegression(random_state=42, penalty='l1', C=1., class_weight='balanced', solver='liblinear').fit(X_train, y_train > thresh)
        with open(params_file, "wb") as f:
            pickle.dump(clf, f)

    y_score_train = clf.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train > 0, y_score_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    print("Train AUC:", roc_auc_train)

    y_score = clf.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test > 0, y_score)
    roc_auc_test = auc(fpr_test, tpr_test)
    print("Test AUC:", roc_auc_test)

    y_score_val = clf.predict_proba(X_val)[:, 1]
    fpr_val, tpr_val, _ = roc_curve(y_val > 0, y_score_val)
    roc_auc_aff3_macs1 = auc(fpr_val, tpr_val)
    print("Aff3/Macs1 AUC:", roc_auc_aff3_macs1)

    X, X_labels, y = kmer_vh_vl_arr, kmer_vh_vl_arr_labels, df["Aff3"].values
    X = pd.DataFrame(X, columns=X_labels)
#     X, y = X[y > 0], y[y > 0]
    y_score = clf.predict_proba(X)[:, 1]
    spearman_aff3 = stats.spearmanr(y_score[y > 0], y[y > 0])[0]
    print("Aff3 Spearman:", spearman_aff3)
    save_df = df.copy()
#     save_df = save_df[save_df["Aff3"] > 0]
    save_df["LR_score"] = y_score
    save_df.to_csv(aff3_file, index=False)

    if len(y_val2) != 0:
        y_score_val2 = clf.predict_proba(X_val2)[:, 1]
        fpr_val2, tpr_val2, _ = roc_curve(y_val2 > 0, y_score_val2)
        roc_auc_spr = auc(fpr_val2, tpr_val2)
        print("SPR AUC:", roc_auc_spr)

    print("LR coefs:")
    coefs = pd.Series(index=clf.feature_names_in_, data=clf.coef_[0])
    print(coefs[coefs != 0].sort_values(ascending=False))
    print(coefs[coefs != 0])

    results.append(dict(
        fname=fname,
        n_params=clf.coef_.shape[1],
        nonzero_params=(clf.coef_ != 0).sum(),
        roc_auc_train=roc_auc_train,
        roc_auc_test=roc_auc_test,
        spearman_aff3=spearman_aff3,
        roc_auc_aff3_macs1=roc_auc_aff3_macs1,
        roc_auc_spr=roc_auc_spr,
        train_size=len(y_train),
        train_positive=(y_train > 0).sum(),
        test_size=len(y_test),
        test_positive=(y_test > 0).sum(),
        aff3_size=len(y_val),
        aff3_positive=(y_val > 0).sum(),
        spr_size=len(y_val2),
        spr_positive=(y_val2 > 0).sum(),
    ))

    # method I: plt
    plt.title(f"{target} ROC")
    plt.plot(fpr_test, tpr_test, 'C0', label = 'Aff1 AUC = %0.2f' % roc_auc_test)
    plt.plot(fpr_val, tpr_val, 'C1', label = 'Aff3 AUC = %0.2f' % roc_auc_aff3_macs1)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f"plots/{target}_test_val_ROC_curve.pdf")
    plt.close()

    fig, ax = plt.subplots()
    plt.title(f"{target} SPR")
    plt.scatter(char_df["SPR KD"].mask(lambda col: col > 300, 300), y_score_val2)
    ax.invert_xaxis()
    plt.ylabel("model score")
    plt.xlabel("SPR KD (nM)")
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xmajorticklabels()]
    labels = ["Fail" if item == "300" else item for item in labels]
    ax.set_xticks(ax.get_xticks(minor=False)[1:-1])
    ax.set_xticklabels(labels[1:-1])
    plt.savefig(f"plots/{target}_val_SPR_scatter.pdf")
    plt.close()

    fig, ax = plt.subplots()
    plt.title(f"{target} Cell Display")
    plt.scatter(char_df["Cell Display EC50"].mask(lambda col: col > 300, 300), y_score_val2)
    plt.gca().invert_xaxis()
    plt.ylabel("model score")
    plt.xlabel("EC50 (nM)")
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xmajorticklabels()]
    labels = ["Fail" if item == "300" else item for item in labels]
    ax.set_xticks(ax.get_xticks(minor=False)[1:-1])
    ax.set_xticklabels(labels[1:-1])
    plt.savefig(f"plots/{target}_val_CellDisplay_scatter.pdf")
    plt.close()


# In[ ]:


results_df = pd.DataFrame(results)
results_df["fname"] = results_df["fname"].str.split("/").str[-1]
results_df


# In[ ]:


results_df[["roc_auc_aff3_macs1", "roc_auc_spr"]].mean()


# # select sequences

# Use the logistic regression model to select sequences from the FACS1 pool according to their read fraction, model score, and distance from other sequences.

# In[ ]:


def min_levenshtein(seq1, seqs):
    return min(distance(seq1, s, score_cutoff=len(seq1)) for s in seqs)

def min_pairwise_levenshtein(seqs):
    prev_cdr3s = [""]
    distances = []
    for cdr3 in seqs:
        distances.append(min_levenshtein(cdr3, prev_cdr3s))
        prev_cdr3s.append(cdr3)
    return distances


# In[ ]:


min_aff1_frac = 1 / 5000
min_lr_score = 0.8
min_dist_to_ordered = 5
min_pairwise_dist = 3


# In[ ]:


results = []

for sort in sorts:
    fname = sort["filename"]
    target = fname.rsplit("/", 1)[1].split("_", 1)[0]
    light_chains = sort["light"]
    heavy_chains = sort["heavy"]
    antigens = sort["SPR_antigens"]
    params_file = f"params/{fname.rsplit('/', 1)[1].replace('.csv', '')}_kmer_LR.pkl"
    aff3_file = f"scores/{fname.rsplit('/', 1)[1].replace('.csv', '')}_kmer_aff3_scores.csv"
    output_file = f"selected_abs/aff1_subset_kmer_LR_{fname.rsplit('/', 1)[1].replace('.csv', '')}.csv"
    if fname not in [
        "zenodo_data/hROBO1_Test1_Lib31.csv",
        "zenodo_data/hROBO2N_Test1_Lib31.csv",
    ]:
        continue

    print(fname)

    df = pd.read_csv(aff3_file, low_memory=False)
    ordered_set = df[df["SEC"].notnull()]

    min_aff1_frac = 1 / 5000
    min_lr_score = 0.8
    min_dist_to_ordered = 5
    min_pairwise_dist = 5 if fname != "zenodo_data/hPD-L2_chase_Test1_Lib31.csv" else 3

    subset_df = df[(df["Aff1"] > df["Aff1"].sum() * min_aff1_frac) & df["SEC"].isnull() & (df["LR_score"] > min_lr_score)].sort_values(by="LR_score", ascending=False)
    subset_df["min_dist_to_ordered"] = subset_df["CDR3"].apply(min_levenshtein, args=(ordered_set["CDR3"],))
    subset_df = subset_df[subset_df["min_dist_to_ordered"] >= min_dist_to_ordered]


    subset_df["min_pairwise_dist"] = min_pairwise_levenshtein(subset_df["CDR3"].values)
    subset_df = subset_df[subset_df["min_pairwise_dist"] >= min_pairwise_dist]
    subset_df.to_csv(output_file, index=False)

    with pd.option_context("display.max_rows", 200):
        print(subset_df)
