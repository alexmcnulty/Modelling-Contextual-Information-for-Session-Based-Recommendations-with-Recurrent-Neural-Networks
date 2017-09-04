
# coding: utf-8

# In[6]:

import numpy as np
import pandas as pd

class RandomPredections:
    """

    RandomPredictions()

    predicts the next domain randomly

    """

    def train(self, data):

        """
        dummy functioon
        :param data:
        :return:
        """
        pass

    def predict_domain(self, session_id, domain_id, domains_to_predict):

        """
        
        :param session_id: Session the domain is in 
        :param domain_id: domain id associated with the domain
        :param domains_to_predict: 
        :return: pandas series that randomly selects the domains
        """

        return pd.Series(data = np.random.rand(len(domains_to_predict)), index = domains_to_predict)

    
class Popular:
    
    def __init__(self, top_n =  10, domain_key = 'Domain_id'):
        self.top_n = top_n
        self.domain_key = domain_key
    
    def train(self,data):
        """
        Get the popularity of each domain in the training set by rank = support(domain)/(support(domain)+1)
        """
        groups = data.groupby(self.domain_key)
        self.popularity = groups.size()
        self.popularity = self.popularity/(self.popularity +1)
        self.popularity.sort_values(ascending = False, inplace = True)
        self.popularity = self.popularity.head(self.top_n)
    
    def predict_domain(self, session_id, domain_id, domains_to_predict):
        
        predictions = np.zeros(len(domains_to_predict))
        # find which domains are in the top 100
        mask = np.in1d(domains_to_predict, self.popularity.index)
        predictions[mask] = self.popularity[domains_to_predict[mask]]
        return pd.Series(data = predictions, index = domains_to_predict)

class SessPopular:
    
    def __init__(self, top_n =  10, domain_key = 'Domain_id'):
        self.top_n = top_n
        self.domain_key = domain_key
    
    def train(self,data):
        """
        Get the popularity of each domain in the training set by rank = support(domain)/(support(domain)+1)
        """
        groups = data.groupby(self.domain_key)
        self.popularity = groups.size()
        self.popularity = self.popularity/(self.popularity +1)
        self.popularity.sort_values(ascending = False, inplace = True)
        self.popularity = self.popularity.head(self.top_n)
        self.prev_session_id = -1
        
    def predict_domain(self, session_id, domain_id, domains_to_predict):
        
        if self.prev_session_id != session_id:
            self.prev_session_id = session_id
            self.pers = dict()
        v = self.pers.get(domain_id)
        if v:
            self.pers[domain_id] = v + 1
        else:
            self.pers[domain_id] = 1
        
        preds = np.zeros(len(domains_to_predict))
        mask = np.in1d(domains_to_predict, self.popularity.index)
        previous_ids = pd.Series(self.pers)
        preds[mask] = self.popularity[domains_to_predict[mask]] 
        mask = np.in1d(domains_to_predict, previous_ids.index)
        preds[mask] += previous_ids[domains_to_predict[mask]]
        return pd.Series(data=preds, index=domains_to_predict)

class domainKNN:
    
    def __init__(self, n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'Session_id', domain_key = 'Domain_id'):
        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.domain_key = domain_key
        self.session_key = session_key 
    
    def train(self, data):
        data.set_index(np.arange(len(data)), inplace = True)
        domainids = data[self.domain_key].unique()
        num_domains = len(domainids)
        data = pd.merge(data, pd.DataFrame({self.domain_key:domainids, 'Domain_idx':np.arange(len(domainids))}), on=self.domain_key, how='inner')
        sessionids = data[self.session_key].unique()
        data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'Session_idx':np.arange(len(sessionids))}), on=self.session_key, how='inner')
        supp = data.groupby('Session_idx').size()
        session_offsets = np.zeros(len(supp)+1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()
        index_by_sessions = data.sort_values('Session_idx').index.values
        supp = data.groupby('Domain_idx').size()
        domain_offsets = np.zeros(num_domains+1, dtype=np.int32)
        domain_offsets[1:] = supp.cumsum()
        index_by_domains = data.sort_values('Domain_idx').index.values
        self.sims = dict()
        for i in range(num_domains):
            iarray = np.zeros(num_domains)
            start = domain_offsets[i]
            end = domain_offsets[i+1]
            for e in index_by_domains[start:end]:
                uidx = data.Session_idx.values[e]
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx+1]
                user_events = index_by_sessions[ustart:uend]
                iarray[data.Domain_idx.values[user_events]] += 1
            iarray[i] = 0
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1-self.n_sims:-1]
            self.sims[domainids[i]] = pd.Series(data=iarray[indices], index=domainids[indices])
    
    def predict_domain(self, session_id, domain_id, domains_to_predict):
        preds = np.zeros(len(domains_to_predict))
        sim_list = self.sims[domain_id]
        mask = np.in1d(domains_to_predict, sim_list.index)
        preds[mask] = sim_list[domains_to_predict[mask]]
        return pd.Series(data=preds, index=domains_to_predict)
    

class BPR:
    
    
    def __init__(self, n_factors = 100, n_iterations = 10, learning_rate = 0.01, lambda_session = 0.0, lambda_domain = 0.0, sigma = 0.05, init_normal = False,  session_key = 'Session_id', domain_key = 'Domain_id'):

        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_session = lambda_session
        self.lambda_domain = lambda_domain
        self.sigma = sigma
        self.init_normal = init_normal
        self.session_key = session_key
        self.domain_key = domain_key
        self.current_session = None
        
    def initialize(self, data):
        self.U = np.random.rand(self.n_sessions, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_sessions, self.n_factors) * self.sigma
        self.I = np.random.rand(self.n_domains, self.n_factors) * 2 * self.sigma - self.sigma if not self.init_normal else np.random.randn(self.n_domains, self.n_factors) * self.sigma
        self.bU = np.zeros(self.n_sessions)
        self.bI = np.zeros(self.n_domains)
        
    def update(self, uidx, p, n):
        uF = np.copy(self.U[uidx,:])
        iF1 = np.copy(self.I[p,:])
        iF2 = np.copy(self.I[n,:])
        sigm = self.sigmoid(iF1.T.dot(uF) - iF2.T.dot(uF) + self.bI[p] - self.bI[n])
        c = 1.0 - sigm
        self.U[uidx,:] += self.learning_rate * (c * (iF1 - iF2) - self.lambda_session * uF)
        self.I[p,:] += self.learning_rate * (c * uF - self.lambda_domain * iF1)
        self.I[n,:] += self.learning_rate * (-c * uF - self.lambda_domain * iF2)
        return np.log(sigm)
    
    def train(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for domain IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, domain_key, time_key properties).
            
        '''
        domainids = data[self.domain_key].unique()
        self.n_domains = len(domainids)
        self.domainidmap = pd.Series(data=np.arange(self.n_domains), index=domainids)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        data = pd.merge(data, pd.DataFrame({self.domain_key:domainids, 'domainIdx':np.arange(self.n_domains)}), on=self.domain_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionIdx':np.arange(self.n_sessions)}), on=self.session_key, how='inner')     
        self.initialize(data)
        for it in range(self.n_iterations):
            c = []
            for e in np.random.permutation(len(data)):
                uidx = data.SessionIdx.values[e]
                iidx = data.domainIdx.values[e]
                iidx2 = data.domainIdx.values[np.random.randint(self.n_domains)]
                err = self.update(uidx, iidx, iidx2)
                c.append(err)
            print(it, np.mean(c))

    
    def predict_domain(self, session_id, input_domain_id, predict_for_domain_ids):
        '''
        Gives predicton scores for a selected set of domains on how likely they be the next domain in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_domain_id : int or string
            The domain ID of the event. Must be in the set of domain IDs of the training set.
        predict_for_domain_ids : 1D array
            IDs of domains for which the network should give prediction scores. Every ID must be in the set of domain IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected domains on how likely to be the next domain of this session. Indexed by the domain IDs.
        
        '''      
        iidx = self.domainidmap[input_domain_id]
        if self.current_session is None or self.current_session != session_id:
            self.current_session = session_id
            self.session = [iidx]
        else:
            self.session.append(iidx)
        uF = self.I[self.session].mean(axis=0)
        iIdxs = self.domainidmap[predict_for_domain_ids]
        return pd.Series(data=self.I[iIdxs].dot(uF) + self.bI[iIdxs], index=predict_for_domain_ids)
             
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


# In[ ]:



