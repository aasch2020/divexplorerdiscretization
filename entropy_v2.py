import sklearn
from sklearn.tree import DecisionTreeClassifier # can remove
from sklearn.tree import _criterion
import numpy as np
# sklearn.tree._criterion

class DivExplorer_Entropy(ClassificationCriterion):
    r"""DivExplorer Entropy impurity criterion.

    Entropy-based criteria used to split instances
    according to the purity of the outcome function.

    Applicable when the measure f has the form of a probability,
    that is,
    when f is defined on the basis of a boolean outcome function o(x):

        fo(S) = k+ / (k+ + k-)
        where k+ = len({x ∈ S | o(x) = T}) and k- = len({x ∈ S | o(x) = F})

    The entropy is then defined as

        H(S,fo) = −fo(S) * log(fo(S)) − (1−fo(S)) * log(1−fo(S))
    """
    def node_impurity(self, metric='fpr', classes, predicted_classes):
	# Calculate the TN, FN, TP, FP for the current node
        tn = self.sum_total[:, 0] - self.sum_left[:, 0]
        fn = self.sum_left[:, 1]
        tp = self.sum_left[:, 1]
        fp = self.sum_total[:, 1] - self.sum_left[:, 1]

	if metric == 'fpr':
		k_plus = fp
		k_minus = tn
	elif metric == 'fnr':
		k_plus = fn
		k_minus = tp

        if not k_plus + k_minus:
            return 0
        f_o = k_plus / (k_plus + k_minus)
        
        # Compute entropy
        if f_o == 0 or f_o == 1:
            entropy = 0
        else:
            entropy = -f_o * np.log2(f_o) - (1 - f_o) * np.log2(1 - f_o)
        
        return entropy

    """Split quality function,
        To favor balanced splits, we weigh
the entropy by the size of the split nodes, as common in
classification trees [20], [21]. Thus, we let the gain of the
split of S into S1, S2 be:
    """
    def split_gain(self, split_left, split_right, outcome_func, d_Val):
        card_S = np.sum(outcome_func(self.samples[self.start:self.end]))
        card_S1 = np.sum(outcome_func(split_left.samples[split_left.start:split_left.end]))
        card_S2 = np.sum(outcome_func(split_right.samples[split_right.start:split_right.end]))
        return card_S*node_impurity(self, outcome_func)/d_Val - (card_S1*node_impurity(split_left, outcome_func)/d_Val + card_S1*node_impurity(split_right, outcome_func)/d_Val))

