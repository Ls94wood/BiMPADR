import os
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from models import Create

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda:1')
    else:
        return torch.device('cpu')

def contoedge(con):
    ed = sp.coo_matrix(con)
    indices=np.vstack((ed.row,ed.col))
    index=torch.from_numpy(indices).long()
    values=torch.from_numpy(ed.data)
    edge_index=torch.sparse_coo_tensor(index,values,ed.shape)
    return edge_index._indices()

path = ''
device = get_default_device()
re_se = {'processed':[],
         'model':[],
         'tre':[],
         'precision':[],
         'auc':[],
         'acc':[]
        }

for set_name in os.listdir(path):
    # Load the data required for the prediction
    se_orign = set_name.split('+')[5]
    
    drug_se = torch.load(path + '/' + set_name + '/' + 'drug_se.pt')
    se_feature = torch.load(path + '/' + set_name + '/' + 'se_feature.pt')
    drug_struc = torch.load(path + '/' + set_name + '/' + 'drug_struc.pt')[0:2,:]
    drug_expr = torch.load(path + '/' + set_name + '/' + 'drug_expr.pt')
    
    edge_index = contoedge(drug_se) 
    new_con = torch.ones(drug_struc.shape[0],se_feature.shape[0])
    index_new = contoedge(new_con)
    
    # The sample data we use here is the complete data set for this study.
    # If you want to predict compounds with other structures, 
    # replace drug_struc and enter index_new that matches the number of your drug structures

    for doc in os.listdir(path + '/' + set_name):
        if doc.find('pkl') != -1:
            # Load the trained model
            model = torch.load(path + '/' + set_name + '/' + doc)
            # Make predictions in the overall data set
            prob = model(edge_index = edge_index.to(device),drug_struc = drug_struc.to(device),drug_expr = drug_expr.to(device),se_struc = se_feature.to(device),pair =index_new.to(device))

            if se_orign == 'zero':
                processed_adrecs = set_name[:-4] + 'adrecs'
                se_feature = torch.load(path + '/' + processed_adrecs + '/se_feature.pt')
            else:
                se_feature = se_feature
            
            # Predictive performance evaluation
            se_tres = [0.5, 0.9, 0.99]
            for se_tre in se_tres:
                se_c = se_expr.copy()
                se_c[se_c <= se_tre] = 0
                se_c[se_c > se_tre] = 1

                auc = roc_auc_score(se_feature.numpy().flatten(), se_c.values.flatten())
                precision = precision_score(se_feature.numpy().flatten(), se_c.values.flatten())
                acc = accuracy_score(se_feature.numpy().flatten(), se_c.values.flatten())
                re_se['processed'].append(set_name)
                re_se['model'].append(doc)
                re_se['precision'].append(precision)
                re_se['auc'].append(auc)
                re_se['acc'].append(acc)

re_se = pd.DataFrame(re_se)
re_se.to_csv(save_name)
