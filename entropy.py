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
    def node_impurity(self, outcome_func):
        
        k_plus = np.sum(outcome_func(self.samples[self.start:self.end]))
        k_minus = np.sum(np.logical_not(outcome_func(self.samples[self.start:self.end])))
        if not k_plus + k_minus:
            return 0
        f_o = k_plus / (k_plus + k_minus)
        
        # Compute entropy
        if f_o == 0 or f_o == 1:
            entropy = 0
        else:
            entropy = -f_o * np.log2(f_o) - (1 - f_o) * np.log2(1 - f_o)
        
        return entropy
