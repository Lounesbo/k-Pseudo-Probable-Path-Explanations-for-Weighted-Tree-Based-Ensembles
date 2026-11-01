import sys
import numpy as np
from pysat.formula import CNF, WCNF
from math import ceil, floor
from tqdm import tqdm
import encoding_CNF as enc
import my_tree as mt
import subprocess
from pysat.examples.rc2 import RC2
from pysat.examples.optux import OptUx
from pysat.card import CardEnc, EncType
import time
from timeout import timeout
from pysat.solvers import Glucose4

import subprocess
import uuid

sys.path.append('../../proto-suff/build')
#import proto_suff

sign = lambda x: int(x > 0) - int(x < 0)

#Creation of my forest
class decision_forest :
    
    def __init__(self, forest) :
        self._nb_class = forest[0].nb_class
        self._nb_features = forest[0].nb_features
        for t in forest :
            if type(t) != mt.decision_tree or t.nb_class != self._nb_class or t.nb_features != self._nb_features :
                raise AssertionError("All the trees must have the same number of input and output")
        self._forest = forest
        self.explainer = None

        bina = {} #TODO : Improve that to take into account the number of appearence
        for t in forest :
            bina.update(t.bina)
        i = 1
        for k in bina.keys():
            bina[k][0] = i
            i += 1
            somme = 0
            for t in forest :
                val = t.bina.get(k,None)
                if val is not None :
                    somme += val[1]
            bina[k][1] = somme
        self._bina = bina
        impossible_inst_clauses = []
        for idx_feature in range(1,self.nb_features+1):
            lits = []
            threshold = []
            nb = 0
            for k in self._bina.keys() :
                if k[0] == idx_feature :
                    lits.append(self._bina[k][0])
                    threshold.append((k[1],nb))
                    nb += 1
            threshold.sort(reverse=True)
            indices = [y for (x,y) in threshold]
            for i in range(len(indices) - 1) :
                impossible_inst_clauses.append([-lits[indices[i]],lits[indices[i+1]]])
        self._threshold_clauses = impossible_inst_clauses
        self._labels = forest[0].labels
            
        
    @property
    def forest(self) :
        return self._forest
    
    @property
    def nb_features(self) :
        return self._nb_features
    
    @property
    def nb_class(self) :
        return self._nb_class
    
    @property
    def bina(self) :
        return self._bina
    
    @bina.setter
    def bina(self, bina) :
        self._bina = bina
        
    
    @property
    def labels(self) :
        return self._labels
    
    @labels.setter
    def labels(self, labels) :
        if len(labels) == len(self.forest[0].root.probabilities):
            self._labels = labels
        else :
            raise ValueError("labels must have the length than the array of probabilities of the root node")
    
    def take_decision(self, instance) :
        nb_votes = np.zeros(self.forest[0].nb_class)
        for t in self.forest :
            nb_votes[t.take_decision(instance)] += 1
        return np.argmax(nb_votes), int(np.max(nb_votes))
    
    def predict(self, instance) :
        choice = self.take_decision(instance)
        return self.labels[choice[0]], choice[1]
    
    def add_tree(self,tree) :
        if type(tree) != mt.decision_tree :
            raise TypeError("Tree isn't a decision_tree")
        else :
            if tree.nb_class != self._nb_class or tree.nb_features != self._nb_features :
                raise AssertionError("All the trees must have the same number of input and output")
            else :
                self._forest.append(tree)
                
    def to_CNF(self, target=None, card_enc="seqCS", tree_enc="comp", card = True, tree = True, threshold_clauses = True) :
        '''
        Generating Carsten-sinz encoding of cardinality constraint for n variable
        
        Parameters :
            card_enc : str; string which say if we will use the (seqCS)uential, the (parCS)allel or the (tot)alizer encoding for the cardinalitry constraint.
            tree_enc : str; string which say if we will encode the tree using a (comp)lementary approach or the (tseytin) one
            card : boolean; If true, we will put the encoding of the cardinaliry constraint
            tree : boolean; If true, we will put the encoding of the tree of the forest
            
        Returns :
            CNF from pysat.formula
        '''
        global_CNF = []
        nb_tree = len(self.forest)
        nb_features = len(self.bina)
        if card :
            X = [i for i in range(1,nb_tree+1)]
            condition = floor(nb_tree/2)+1
            if card_enc == "seqCS" :
                #cardinality_CNF = enc.my_CNF_card(X,ceil((nb_tree/2)-1)) # /2 because work only with a monoclassification
                cardinality_CNF = CardEnc.atleast(lits=X, encoding=EncType.seqcounter, bound=condition).clauses
            elif card_enc == "tot" :
                #cardinality_CNF = enc.comparator([i for i in range(1,nb_tree+1)],condition)
                cardinality_CNF = CardEnc.atleast(lits=X, encoding=EncType.totalizer, bound=condition).clauses
            else :
                raise ValueError(f"{card_enc} isn't a valid option, you must put card_enc = 'parCS', card_enc = 'seqCS' or card_enc = 'tot'")
            for i in range(len(cardinality_CNF)) :
                clause = cardinality_CNF[i].copy()
                #new_clause = [l+sign(l)*nb_features for l in clause if abs(l) <= nb_tree] + [l + sign(l)*(nb_tree + nb_features) for l in clause if abs(l) > nb_tree]
                new_clause = [l+sign(l)*nb_features for l in clause]
                cardinality_CNF[i] = new_clause
            #add the equivalence of each variable representing a tree to his CNF
            global_CNF.extend(cardinality_CNF.copy())
        if tree :
            if card :
                aux = np.max([np.max([abs(i) for i in clause]) for clause in global_CNF]) + 1
            else :
                aux = None
            for l in range(1+nb_features,1+nb_features+nb_tree) :
                tree = self.forest[l-(nb_features+1)]
                if tree_enc == "comp" :
                    CNF_l = tree.to_CNF(hash_bin=self.bina, target = target, threshold_clauses=False)[0].clauses.copy()
                    CNF_non_l = tree.to_CNF(hash_bin=self.bina, target = target, method = "anti", threshold_clauses=False)[0].clauses.copy()
                elif tree_enc == "tseytin" :
                    CNF_aux = tree.to_CNF(hash_bin=self.bina, method = "tseytin", aux = aux, target = target, threshold_clauses=False) #Transform the DNF -> CNF using Tseytin
                    CNF_l, aux = CNF_aux[0].clauses.copy(), CNF_aux[1]
                    CNF_non_l = tree.to_CNF(hash_bin=self.bina, target = target, threshold_clauses=False)[0].clauses.copy() # Take the negation from the original DNF
                elif tree_enc == "forMUS" :
                    CNF_l = tree.to_CNF(hash_bin=self.bina, target = target, method = "anti", threshold_clauses=False)[0].clauses.copy()
                elif tree_enc == "cfact" :
                    CNF_l = tree.to_CNF(hash_bin=self.bina, target = target, method = "anti", threshold_clauses=False)[0].clauses.copy()
                    CNF_non_l = tree.to_CNF(hash_bin=self.bina, target = target, method = "comp", threshold_clauses=False)[0].clauses.copy()
                elif tree_enc == "compS" :
                    CNF_l = tree.to_CNF(hash_bin=self.bina, target = target, threshold_clauses=False)[0].clauses.copy()
                else :
                    raise ValueError(f"{tree_enc} isn't a valid option, you must put tree_enc = 'comp' or tree_enc = 'tseytin'")
                for c in CNF_l :
                    c.append(-l)
                if tree_enc != "forMUS" and tree_enc != "compS" :
                    for c in CNF_non_l :
                        c.append(l)
                    global_CNF += CNF_l + CNF_non_l
                else :
                    global_CNF += CNF_l
        for l in range(len(global_CNF)) :
            for i in range(len(global_CNF[l])) :
                global_CNF[l][i] = int(global_CNF[l][i])
        if threshold_clauses :
            global_CNF += self._threshold_clauses
        return CNF(from_clauses=global_CNF)
    
    def erase_attribute(self, attribute) :
        for t in self.forest :
            t.erase_attribute(attribute)
            
    def find_sufficient_reason(self, instance, implicant=None , target = None, name=None) :
        '''
        Find a sufficient reason of the feature using the forest structure

        Parameters :
            instance : numpy array or list; represent a binarized feature ex  : [0,1,0,0,1.......,0,1,1]

        Returns :
            List containing a sufficient reason (Conjonction  of litterals)
        '''
        if name is None :
            name = "data"
        if implicant == None :
            bin_instance = self.binarized_instance(instance) #soft clauses
        else :
            bin_instance = implicant
        if target == None :
            target, score = self.take_decision(instance)
        else :
            target = target
        hard_clause = self.to_CNF(target=target, card_enc="seqCS", tree_enc="forMUS", card = True, tree = True, threshold_clauses = False) # hard clauses
        # Writing gcnf file
        fichier = open(f"{name}.gcnf", "w")
        fichier.write(f"p gcnf {hard_clause.nv} {len(hard_clause.clauses) + len(bin_instance)} {len(bin_instance) + 1}\n")
        for c in hard_clause.clauses :
            line = "{1}"
            for l in c :
                line += ' ' + str(l)
            fichier.write(line + ' 0\n')
        i = 2
        for l in bin_instance :
            line = "{" + str(i) + "} " + str(l) 
            i += 1
            fichier.write(line + ' 0\n')
        fichier.close()
        reason_str = subprocess.getoutput(f'../bin/muser2 -comp  -grp {name}.gcnf')
        for line in reason_str.split('\n') :
            if line[0] == "v" :
                reason_str = line
        reason = []
        for txt in reason_str.split(' ')[2:-1] :
            try :
                reason.append(bin_instance[int(txt)-2])
            except IndexError :
                print(name)
                print(txt)
                print(bin_instance)
                print(txt)
            except ValueError:
                print(name)
                print(txt)
                print(bin_instance)
                print(txt)
        return reason
    
    def find_minimal_reason(self, instance, implicant=None , target = None, name=None) :
        '''
        Find a sufficient reason of the feature using the forest structure

        Parameters :
            instance : numpy array or list; represent a binarized feature ex  : [0,1,0,0,1.......,0,1,1]

        Returns :
            List containing a sufficient reason (Conjonction  of litterals)
        '''
        if name is None :
            name = "data"
        if implicant == None :
            bin_instance = self.binarized_instance(instance) #soft clauses
        else :
            bin_instance = implicant
        if target == None :
            target, score = self.take_decision(instance)
        else :
            target = target
        hard_clause = self.to_CNF(target=target, card_enc="seqCS", tree_enc="forMUS", card = True, tree = True, threshold_clauses = False) # hard clauses
        CNF_min = WCNF()
        for hc in hard_clause.clauses :
            CNF_min.append(hc)
        for l in bin_instance :
            CNF_min.append([l], weight=1)
        mini = len(bin_instance)
        reason = bin_instance
        with OptUx(CNF_min) as optux :
            for mus in optux.enumerate() :
                if len(mus) < mini :
                    reason = mus
                    mini = len(mus)
        return reason

    def enumerate_majoritary_reasons(self, instance, implicant=None, target=None, time_out=36000000, nb=1, excludedFeatures = None) :
        if implicant == None :
            bin_instance = self.binarized_instance(instance)
        else :
            bin_instance = implicant
        if target == None :
            target, score = self.take_decision(instance)
        else :
            target = target
        CNF_forest = self.to_CNF(target=target, card_enc="seqCS", tree_enc="compS", threshold_clauses = False).clauses

        maxlit = max([abs(l) for l in bin_instance])
        cnf = CNF()
        for c in CNF_forest:
            cnf.append([l for l in c if l in bin_instance or abs(l) > maxlit])

        glucose = Glucose4()
        for cl in cnf:
            glucose.add_clause(cl)

        # excluded features are forbidden
        if excludedFeatures is not None:
            tmp = self.extract_excluded_features(bin_instance, excludedFeatures)
            for l in tmp:
                glucose.add_clause([-l])

        majoritaries = []
        while True:
            result = glucose.solve()
            if result == False or (nb != -1 and len(majoritaries) >= nb):
                break
            model = glucose.get_model()
            maj = [l for l in model if l in bin_instance]
            if maj not in majoritaries:
                majoritaries.append(maj)
            # block this implicant
            glucose.add_clause([-l for l in model if abs(l) <= maxlit])

        return majoritaries


    def find_pseudo_min_reason(self, instance, time_out=36000000, nb=1) :
        return self.find_prefered_majoritary_reason(instance, method=None, time_out = time_out, nb = nb)
        
    
    def find_direct_reason(self, instance) :
        target, score = self.take_decision((instance))
        direct_reason = []
        for t in self.forest :
            if t.take_decision(instance) == target :
                direct_reason += t.find_direct_reason(instance, hash_bin = self._bina)
        return list(set(direct_reason))
  

    def find_prefered_majoritary_reason(self, instance, *, data=None, rf = None, col = None, w_u = None , method = 'shapely',
                                        solver=None, nb = 1, excludedFeatures = None, time_out=36000000):
    
        bin_instance = self.binarized_instance(instance)
        target, score2 = self.take_decision(instance)
        cnf = self.to_CNF(target=target, card_enc="seqCS", tree_enc="compS", threshold_clauses=False)
        CNF_forest = cnf.clauses
        nb_vars = cnf.nv
        w_p = self.unbinarized_instance(bin_instance)

    
        if method == 'shapely' : # Shapely poids
            explainer = shap.TreeExplainer(rf, data=data, model_output='raw') # Shap value for rf model trained on data.
            shap_values = explainer.shap_values(instance) # shapely values
            if len(shap_values) > 2 : # if multi-classes classification  
                shap_value = np.mean(shap_values, axis=0)
            else : shap_value = shap_values[0] # binary classification 
            
            # decreasing monotonous affine transformation for shap values, w = alpha * x + b , where x = shap_value
            a = min([i for i in abs(shap_value) if i != 0]) 
            shap_value = -shap_value/min(a,1) # alpha = - min(a, 1) where a is the minimum value greater than zero of abs(shap_value)
            w = [round(shap_value[i] - min(shap_value) + 1) for i in range(len(shap_value))] #  b = - min(alpha*shap_value) + 1


        elif method == 'f_importance' : # features importance from sklearn
            # decreasing monotonous affine transformation
            x = rf.feature_importances_
            a = min([i for i in x if i != 0]) # the minimum value greater than zero
            x = -x/a # alpha = -10/a, b = 1
            w = [round(x[i] - min(x) + 1) for i in range(len(x))]
            # Generate our clauses


        elif method == 'wordfreq' : # zipf_frequency "wordfreq package" with (decreasing)
            w_c = [int(zipf_frequency(l, 'en')*100*(zipf_frequency(l, 'en') > 0) + (zipf_frequency(l, 'en') == 0)) for l in col]
            w = [-w_c[i] + 1 + max(w_c) for i in range(len(w_c))] 


        elif method == 'strate_wordfreq':
            w = [0 for i in range(len(col))]
            a = [0 for i in range(3)] # number of strate - 1
            for i in range(len(col)):
                if 6 < zipf_frequency(col[i], 'en') :
                    a[0] += 1
                    w[i] = 1
                if 4 < zipf_frequency(col[i], 'en') <= 6:
                    a[1] += 1
                    w[i] = 10 + a[0] # for non-compensation 
                if 2 < zipf_frequency(col[i], 'en') <= 4:
                    a[2] += 1
                    w[i] = 20 + a[0] + 10*a[1] # for non-compensation
                if 0 <= zipf_frequency(col[i], 'en') <= 2:
                    w[i] =  100 + a[0] + 10*a[1] + 20*a[2] # for non-compensation


        elif method == 'user' : # w_u : the weights chosen by the user, which correspond to the user's preference
            w = [-w_u[i] + 1 + max(w_u) for i in range(len(w_u))]


        elif method == None:
            w = [1 for l in range(len(instance))]
        # start to create a formula

        wcnf = WCNF() # WCNF from pysat
        maxlit = max([abs(l) for l in bin_instance]) 
        # hard clauses (cnf clauses)
        in_bin_instance = [0 for i in range(0, nb_vars + 1)]
        for l in bin_instance:
            in_bin_instance[abs(l)] = l

        for c in CNF_forest:
            wcnf.append([l for l in c if abs(l) > maxlit or in_bin_instance[abs(l)] == l])

        # excluded features are forbidden
        if excludedFeatures is not None:
            tmp = self.extract_excluded_features(bin_instance, excludedFeatures)
            for l in tmp:
                wcnf.append([-l])


        # soft clause 
        for i in range(len(bin_instance)):
            wcnf.append([-bin_instance[i]], weight = w[w_p[i]-1]) 

        first = True
        results = []
        w_best = -1

        if solver == "openwbo":
            #assert(nb == 1)
            ###########################
            time_used = 0
            results = []
            first = True
            #############################
            while time_used < time_out :
                hash = str(uuid.uuid4().fields[-1])[:8]
                file = f"/tmp/wbo-{hash}.wcnf"
                wcnf.to_file(file)
                tps = -time.time()
                p = subprocess.run(["../bin/open-wbo", f"-cpu-lim={time_out-time_used}", file], capture_output=True, text=True)
                tps += time.time()
                time_used += tps
                output_str = p.stdout
    
                m = [line.split(" ") for line in output_str.split("\n") if len(line) > 0 and line[0] == "v"]
                if len(m) > 0:
                    m = m[0]
                    model = [int(l) for l in m if l != 'v' and l != '']
                    ##################################################
                    prefered = [l for l in model if l in bin_instance]
                    wcnf.append([-l for l in model if abs(l) <= maxlit])
                    score = np.sum([w[w_p[abs(l)-1]-1] for l in prefered])
                    if nb == 1:
                        return prefered
                    if first :
                        w_best = score
                    else:
                        if score != w_best:
                            return results
                    first = False
                    ######################################################
                    status = [line.split(" ")[1] for line in output_str.split("\n") if len(line) > 0 and line[0] == "s"][0]
                    if status == "OPTIMUM":
                        results.append(prefered)
                    else:
                        implicant =  [l for l in model if l in bin_instance]
                        results.append([0]+self.find_majoritary_reason(instance, implicant = implicant)) #Le "0" au début est pour signifier que ce n'est pas une vrai raison mais une approximation
                else:
                    return results
        else :
            nb_computed = 0
            with RC2(wcnf, incr=True, minz=True) as rc2:
                t_ref = -time.time()
                while t_ref + time.time()< time_out and len(results) < nb :
                    result = rc2.compute()
                    if result == None:
                        return results
                    score = rc2.cost
                    prefered = [l for l in result if l in bin_instance]
                    rc2.add_clause([-l for l in result if abs(l) <= maxlit])
                    if nb == 1:
                        return prefered
                    if first :
                        w_best = score
                    else:
                        if score != w_best:
                            return results
                    first = False
                    if prefered not in results :
                        results.append(prefered)  # prefered majoritary reason
                    nb_computed += 1
        return results
    '''=======
            assert(nb == 1)
            hash = str(uuid.uuid4().fields[-1])[:8]
            file = f"/tmp/wbo-{hash}.wcnf"
            wcnf.to_file(file)
            p = subprocess.run(["../bin/open-wbo", f"-cpu-lim={time_out}", file], capture_output=True, text=True)
            output_str = p.stdout

            m = [line.split(" ") for line in output_str.split("\n") if len(line) > 0 and line[0] == "v"]
            if len(m) > 0:
                m = m[0]
                model = [int(l) for l in m if l != 'v' and l != '']
                status = [line.split(" ")[1] for line in output_str.split("\n") if len(line) > 0 and line[0] == "s"][0]
                if status == "OPTIMUM":
                    return ['OPTIMUM', [l for l in model if l in bin_instance]]
                else:
                    implicant =  [l for l in model if l in bin_instance]
                    return ['APPROX' , self.find_majoritary_reason(instance, implicant = implicant)]
            else:
                return []
>>>>>>> 53f7daa1625276c0fbb2be591052a2c35d64ce68'''



    def binarized_instance(self, instance, pref_order=None, hash_bin = None) :
        """
        Binarized an instance according to the binarization of the tree

        Parameters :
            instance: a list/numpy array representing the feature's values of an instance
            
        Returns:
            A list corresponding to the binarized instance
        """
        if hash_bin is None :
            hash_bin = self._bina
        output = [] 
        for k in hash_bin.keys() :
            if instance[k[0]-1] > k[1] :
                output.append(hash_bin[k][0])
            else :
                output.append(-hash_bin[k][0])
        return output
                        
            
    
    def unredundant_binarized_instance(self, instance_bin) :
        """
        Binarized an instance according to the binarization of the tree

        Parameters :
            instance_bin: a list/numpy array representing the instance in his binary version
            
        Returns:
            A list corresponding to the binarized instance without redundant information i.e. feature a > 3 and feature_a > 2, we keep only the int linked too the boolean corresponding to feature_a > 3
        """
        pair_pos = {}
        pair_neg = {}
        for k in self._bina.keys() :
            if self.bina[k][0] in instance_bin:
                pair_pos[k[0]] = k[1]
            if -self.bina[k][0]  in instance_bin:
                pair_neg[k[0]] = k[1]
        output = []
        for k in pair_pos.keys():
            output.append(self.bina[(k,np.max(pair_pos[k]))][0])
        for k in pair_neg.keys():
            output.append(-self.bina[(k,np.min(pair_neg[k]))][0])
        return output
    
    def unbinarized_instance(self, instance_bin, need_detail = False) :
        """
        Binarized an instance according to the binarization of the tree

        Parameters :
            instance_bin: a list/numpy array representing the instance in his binary version
            
        Returns:
            A list corresponding to the indices of the original attributes present in instance_bin   
        """
        output = []
        if not need_detail :
            for k in self._bina.keys() :
                if (self.bina[k][0] in instance_bin) or (-self.bina[k][0]  in instance_bin) :
                    output.append(k[0])
            return output
        else :
            instance_bin = self.unredundant_binarized_instance(instance_bin)
            for k in self._bina.keys() :
                if (self.bina[k][0] in instance_bin) :
                    output.append([k[0], k[1], "+"])
                elif (-self.bina[k][0]  in instance_bin) :
                    output.append([k[0], k[1], "-"])
            return output
        
    def generate_instance(self, reason, hash_bin = None) :
        """
        WARNING : Work ONLY with mnist dataset or a declinaison of it
        
        Create an instance matching with a known reason
        
        Parameters :
            reason : a list of int descvribing a reason ex : [-6,9,23,-25,42]
        
        Returns :
            A list corresponding to an instance that this forest can manage
        """
        if hash_bin is None :
            hash_bin = self._bina
        reverse_hash_bin = {}
        for k in hash_bin.keys() :
            reverse_hash_bin[hash_bin[k][0]] = k
        instance = [int(np.random.rand(1)[0]*255) for i in range(self.nb_features)]
        exp = self.unbinarized_instance(reason, need_detail = True)
        for e in exp :
            if e[2] == "+" :
                instance[e[0]-1] = e[1] + 1
            else :
                instance[e[0]-1] = e[1] - 1
        return instance
    
    def compileForest(self, target=None, write_file = False, name_file = "test.txt") :
        """
        Generate a list of list of int to describe the forest

        Parameters
        ----------
        target : class we need 

        Returns
        -------
        output : List of list of int.

        """
        if target == None :
            target = 0
        output = []
        for t in self.forest :
            output.append(t.compileTree(hash_bin = self.bina, target=target))
        if write_file :
            file = open(name_file, "w")
            for t_txt in output :
                file.write(str(t_txt)[1:-1].replace(",","")+"\n")
            file.close()
        return output


    '''def init_explainer(self):
        self.explainer = proto_suff.MinimumMajoritaryExplainer()
        fc = self.compileForest(0)

        for tree in fc:
            self.explainer.addTree(tree)'''



    def find_majoritary_reason(self, instance, implicant = None, excludedFeatures = None) :
        if self.explainer is None:
            self.init_explainer()
        if implicant is None :
            bin_instance = self.binarized_instance(instance)
        else:
            bin_instance = implicant

        if excludedFeatures is not None:
            excluded = self.extract_excluded_features(bin_instance, excludedFeatures)
            return self.explainer.explainWithExcludedFeatures(bin_instance, excluded)
        else :
            return self.explainer.explain(bin_instance)


    def is_a_majority_implicant(self, reason, target, hash_bin=None) :
        """
        Say if reason is a majority implicant of the forest
        
        Parameters :
            reason : list of int; a list describing a potential implicant
            target : class for which reason is potentially an implicant
            hash_bin : dict; A dict corresponding to a hashmap (num_feature, threshold) <-> (associated boolean, number of appearence) NB : You can use a dict saying only the associated boolean
            
        Returns :
            A boolean
        """
        if hash_bin is None :
            hash_bin = self._bina
        nb_impl = 0
        for t in self.forest :
            CNF_tree = t.to_CNF(target=target, hash_bin=hash_bin, method="comp", threshold_clauses=False)[0].clauses
            implicated = True
            for c in CNF_tree :
                clause_implicated = False
                for l in reason :
                    if l in c :
                        clause_implicated = True
                if not clause_implicated :
                    implicated = False
            if implicated :
                nb_impl += 1
        if nb_impl >= (len(self.forest)//2) + 1 :
            return True
        else :
            return False
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
