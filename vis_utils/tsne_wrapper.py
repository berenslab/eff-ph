from openTSNE import TSNE
from openTSNE.callbacks import Callback
from .utils import KL_divergence, compute_normalization
import numpy as np
import time

class Logger(Callback):
    """
    Computes and logs quantities of interest during an embedding optimization
    """
    def __init__(self,
                 log_kl=True,
                 log_embds=False,
                 log_Z=True,
                 log_time=True):
        """
        Initialization
        :param log_kl: bool If True, log the KL divergence
        :param log_embds: bool If True, log intermediate embeddings
        :param log_Z: bool If True, log partition function
        """
        super().__init__()
        self.errors = []

        self.log_kl = log_kl
        if self.log_kl:
            self.kl_divs = []

        self.log_embds = log_embds
        if self.log_embds:
            self.embeddings = []

        self.log_time = log_time

        self.log_Z = log_Z
        if self.log_Z:
            self.Zs = []
    def __call__(self, iteration, error, embedding):
        self.errors.append(error)
        if self.log_embds:
            self.embeddings.append(np.array(embedding))
        if self.log_kl:
            self.kl_divs.append(KL_divergence(embedding.affinities.P.tocoo(),
                                              embedding=np.array(embedding),
                                              a=1.0,
                                              b=1.0,
                                              norm_over_pos=False,
                                              eps=np.finfo(np.float64).eps))
        if self.log_Z:
            self.Zs.append(compute_normalization(np.array(embedding),
                                                 sim_func="cauchy",
                                                 no_diag=True,
                                                 a=1.0,
                                                 b=1.0,
                                                 eps=float(np.finfo(float).eps)))
        return False

class TSNEwrapper:
    """
    Wrapper to the TSNE class that add logging
    """
    def __init__(self, log_kl=True, log_embds=True, log_Z=True, log_time=True, **tsne_kwargs):
        self.logger = Logger(log_kl, log_embds, log_Z, log_time)
        self.tsne = TSNE(callbacks=[self.logger],
                         **tsne_kwargs)
        self.aux_data = tsne_kwargs
        self.aux_data["log_kl"] = log_kl

    def fit_transform(self, X=None, affinities=None, initialization=None):
        # compute tsne
        if self.logger.log_time:
            start_time = time.time()
        embd = self.tsne.fit(X=X,
                             affinities=affinities,
                             initialization=initialization)


        # add the logs to aux_data
        self.aux_data["errors"] = np.array(self.logger.errors)
        self.aux_data["graph"] = embd.affinities.P.tocoo()
        self.aux_data["init"] = initialization


        if self.logger.log_embds:
            self.aux_data["embds"] = np.stack(self.logger.embeddings, axis=0)
        else:
            self.aux_data["embd"] = np.array(embd)
        if self.logger.log_kl:
            self.aux_data["kl_div"] = np.array(self.logger.kl_divs)
        if self.logger.log_Z:
            self.aux_data["Zs"] = np.array(self.logger.Zs)

        if self.logger.log_time:
            self.aux_data["time"] = time.time() - start_time
        return embd