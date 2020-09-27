from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score,fbeta_score
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import roc_curve,auc,roc_auc_score
import matplotlib.pyplot as plt


class InspectorStandard():
    def __init__(self,acc_normalize=True,p_average='micro',gotNames=False,
                 r_average='micro',f1_average='micro',fb_average='micro',beta=1,
                 pr_average='micro',roc_average='macro'):
        self.parameters = {}
        self.parameters['gotNames'] = gotNames
        self.parameters['acc_normalize'] = acc_normalize
        self.parameters['p_average'] = p_average
        self.parameters['r_average'] = r_average
        self.parameters['f1_average'] = f1_average
        self.parameters['fb_average'] = fb_average
        self.parameters['beta'] = beta
        self.parameters['pr_average'] = pr_average
        self.parameters['roc_average'] = roc_average

    def inspect(self,y_true,y_pred,y_logit=None):
        self.n_classes = len(np.unique(y_true))
        self.n_samples = len(y_true)
        results = {}

        #These parameters are identically computed in bi and multi class scenario
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred, normalize=self.parameters['acc_normalize'])
        kappa = cohen_kappa_score(y_true, y_pred)
        results['cm'] = cm ;results['acc'] = acc;results['kappa'] = kappa;results['error']=1-acc

        #These parameters have different extension methods to multi class
        p = precision_score(y_true, y_pred, average=self.parameters['p_average'])
        r = recall_score(y_true, y_pred, average=self.parameters['r_average'])
        f1 = f1_score(y_true, y_pred, average=self.parameters['f1_average'])
        fbeta = fbeta_score(y_true,y_pred,beta=self.parameters['beta'],average=self.parameters['fb_average'])
        results['p'] = p;results['r'] = r;results['f1'] = f1;results['fbeta'] = fbeta

        if type(y_logit)==np.ndarray:
            y_one_hot = label_binarize(y_true, np.arange(self.n_classes))
            if self.parameters['pr_average'] == 'micro':
                p_curve, r_curve, th_pr = precision_recall_curve(y_one_hot.ravel(), y_logit.ravel())
                average_precision = average_precision_score(y_one_hot, y_logit, average='micro')
            elif self.parameters['pr_average'] == 'macro':
                p_curve, r_curve ,th_pr = [],[],[]
                ap = 0
                for i in range(self.n_classes):
                    y_logit_i = y_logit[:, i]
                    y_one_hot_i = y_one_hot[:, i]
                    p_curve_i, r_curve_i, th_pr_i = precision_recall_curve(y_one_hot_i, y_logit_i)
                    # 注意这里每次算出的fpr_i和tpr_i形状都有可能不同
                    p_curve.append(p_curve_i);r_curve.append(r_curve_i);th_pr.append(th_pr_i)
                    ap_i = average_precision_score(y_one_hot_i, y_logit_i)
                    ap += ap_i
                ap /= y_one_hot.shape[1] #ap==average_precision
                average_precision = average_precision_score(y_one_hot, y_logit, average='macro')
            results['p_curve'] = p_curve;results['r_curve'] = r_curve
            results['th_pr'] = th_pr;results['average_precision'] = average_precision


            if self.parameters['roc_average'] == 'micro':
                fpr_curve, tpr_curve, th_roc = roc_curve(y_one_hot.ravel(), y_logit.ravel())
                roc_auc = roc_auc_score(y_one_hot,y_logit,average='micro')
            elif self.parameters['roc_average'] == 'macro':
                fpr_curve, tpr_curve, th_roc = [], [], []
                roc_auc = 0
                for i in range(y_one_hot.shape[1]):
                    y_logit_i = y_logit[:, i]
                    y_one_hot_i = y_one_hot[:, i]
                    fpr_i, tpr_i, th_roc_i = roc_curve(y_one_hot_i, y_logit_i)
                    # 注意这里每次算出的fpr_i和tpr_i形状都有可能不同
                    fpr_curve.append(fpr_i);tpr_curve.append(tpr_i);th_roc.append(th_roc_i)
                    ac_i = auc(fpr_i, tpr_i)
                    roc_auc += ac_i
                roc_auc /= self.n_classes
            results['fpr_curve'] = fpr_curve;results['tpr_curve'] = tpr_curve
            results['th_roc'] = th_roc;results['roc_auc'] = roc_auc

        del self.n_classes,self.n_samples
        return results


"""
example

y_true = np.array([0, 1, 2, 2, 0 ,1, 0])
y_pred = np.array([0, 0, 2, 2, 0 ,1, 1])
y_logit = np.array([[0.4,0.3,0.3],
           [0.6,0.1,0.3],
           [0.1,0.3,0.6],
           [0.2,0.1,0.7],
           [0.6,0.2,0.2],
           [0.2,0.5,0.3],
           [0.1,0.8,0.1]])
Is = InspectorStandard()
re = Is.inspect(y_true,y_pred,y_logit)

"""
