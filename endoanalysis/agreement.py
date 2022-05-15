import os
import itertools
import copy
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import pingouin as pg
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm as normal_distribution
from sklearn.metrics import cohen_kappa_score
from endoanalysis.datasets import PointsDataset

    

def get_batch_relaibility_matrix(
    targets_from_batch1, 
    targets_from_batch2, 
    similarity,
    indexes_to_consider="all",
    drop_missed = False
):
    '''
    Composes relaibility matrix for two batches.
    
    Parameters
    ----------
    targets_from_batch1 : endoanalysis.targets.ImageTargetsBatch like
        batch of targets corresponding to a set of images from the first rater in pair. 
        Must have have the methods targets_from_batch1.from_image() and targets_from_batch1.num_images(),
        returning the container of targets corrsponding to a given image.
        The container must be compatible with similarity.
    targets_from_batch2 :  endoanalysis.targets.ImageTargetsBatch  like
        the same as targets_from_batch1, but for the second rater.
    similarity : endoanalysis.agreement.SimilarityMeasure
        similarity measure, must be compatible with targets_from_batch1 and targets_from_batch2
    indexes_to_consider : iterable or str
        an iterable of image indexes which sould be taken into concideration. 
        All indexes must be present in both  targets_from_batch1 an targets_from_batch2.
        If indexes_to_consider == "all", all the present images will be processed
    drop_missed : bool
        wheather to consider not matched targeds as separate class (with -1 label)
        
    Returns
    -------
    relaibility_matrix : ndarray
        relaibility matrix. The shape is (2, num_matched), where num_matched
        is the number of targets which were successfully matched. This number
        could not be greater than the maximum total number of targets in
        batch1 or batch2
    '''
    
    rel_matrices = []
    
    if indexes_to_consider == "all":
        len1 = targets_from_batch1.num_images()
        len2 = targets_from_batch2.num_images()
        if len1 != len2:
            raise Exception("Different number of images in batches for agreement computation: %i, %i"%(len1, len2))
        else:
            indexes_to_consider = range(len1)
        
    for image_i in indexes_to_consider:
        targets1 = targets_from_batch1.from_image(image_i)
        targets2 = targets_from_batch2.from_image(image_i)

        sim_matrix = similarity.matrix(targets1, targets2)

        row_ids, col_ids = linear_sum_assignment(sim_matrix, maximize=True)
        
        rel_matrix = np.vstack([targets1.classes()[row_ids], targets2.classes()[col_ids]])
#         rel_matrix = np.ones_like(rel_matrix)
        rel_matrices.append(rel_matrix)

        
        if not drop_missed:
            missings1 = np.setdiff1d(np.arange(len(targets1)), row_ids) 
            missings2 = np.setdiff1d(np.arange(len(targets2)), col_ids) 


            rel_matrices.append(np.vstack([targets1.classes()[missings1], np.ones(len(missings1)) * -1]))
            rel_matrices.append(np.vstack([np.ones(len(missings2)) * -1, targets2.classes()[missings2] ]))

    relaibility_matrix = np.hstack(rel_matrices)
    
    return relaibility_matrix


def compute_kappas(
    targets_containers,
    similarity,
    experts_list,
    indexes_to_consider="all",
    drop_missed=False,
    sig_level = 0.05,
    compute_all_kappas=False,
):
    '''
    Compute Cohen's kappa for all raters from raters_list.  
    
    Parameters
    ---------
    targets_containers : dict of endoanalysis.targets.ImageTargetsBatch
        a dict with batches of targets. The keys are the raters ids.
        ImageTargets batches should be compatible with similarity
    similarity : endoanalysis.agreement.SimilarityMeasure
        similarity measure, must be compatible with targets_from_batch1 and targets_from_batch2
    experts_list : iterable
        an iterable of experts to consider. All listed experts must be in targets_containers keys
    indexes_to_consider : iterable
        an iterable of image indexes which sould be taken into concideration. 
        All indexes must be present in all targets_containers.
        If indexes_to_consider == "all", all the present images will be processed
    drop_missed : bool
        wheather to consider not matched targeds as separate class (with -1 label)
    sig_level: float
        significance level for the confidence interval
    compute_all_kappas : bool
        Usful for debug purposes. If True, kappas scores will be computed for all experts pairs, even if the pairs are differ by permutation or contains equal indexes. (So the pairs (1,2) and (2,1) will be computed separately, and the pairs (1,1) and (2,2) will also be computed). 
        If False, only lower offdiagonal pairs are actually computed, while diagonal pairs are hardcoded to be equal to 1.
   
    Returns
    -------
    kappas : ndarray
        the matrix of pairwise kappas. kappas[i,j] is a Cohen kappa for experts_list[i] and experts_list[j]. kappas[i,i] are always equals to 1,
    deltas: ndarray
        the matrix of pairwise confidence interval half lengths for corresponding kappas
    experts_to_ids: dict
        mapping expert's names to their ids in kappas matrix
    '''
    
    num_experts = len(experts_list)
    kappas = np.zeros((num_experts, num_experts))
    deltas = np.zeros((num_experts, num_experts))
    experts_to_ids = {y:x for x, y in enumerate(experts_list)}
    
    
    if compute_all_kappas:
        iterator = itertools.product(experts_list, 2)
    else:
        iterator = itertools.combinations(experts_list, 2)
        
    for expert_1, expert_2 in iterator:
        relaibility_matrix = get_batch_relaibility_matrix(
            targets_containers[expert_1],
            targets_containers[expert_2],
            similarity,
            indexes_to_consider,
            drop_missed=drop_missed
        )
        kappa, delta = kappa_score(relaibility_matrix[0], relaibility_matrix[1], sig_level)

        for cont, value in zip([kappas, deltas], [kappa, delta]):
            cont[
                experts_to_ids[expert_1], 
                experts_to_ids[expert_2]
            ] = value

    
    if not compute_all_kappas:
        kappas += kappas.transpose()
        deltas += deltas.transpose()
        for i in range(num_experts):
            kappas[i,i] = 1.
            
    return kappas, deltas, experts_to_ids

def kappa_score(choices_1, choices_2, sig_level = 0.05):
    '''
    Compute Cohen's cappa and it's confidence inteval at a given significance level.

    Parameters
    ----------
    choices_1: ndarray
        choices of the first expert
    choices_2: ndarray
        choices of the second expert. Must be the same length as choices_first
    sig_level: float
        significance level for the confidence interval

    Returns
    -------
    kappa: float
        Cohen's kappa
    delta: float
        confidence interval half length

    See also
    --------
    The method for computing kappa's standart deviation see
    Fleiss, Cohen, Everitt(1969). Large sample standard errors of kappa and weighted kappa.
    https://doi.org/10.1037/h0028106
    '''

    if len(choices_1) != len(choices_2):
        raise Exception("Dirst and second experts must make the same number of choices.")

    z = normal_distribution.ppf(1 - sig_level / 2)
    values = np.unique(np.concatenate([choices_1, choices_2]))
    num_samples = len(choices_1)

    first_rater = choices_1.reshape(1,-1) == values.reshape(-1,1)
    second_rater = choices_2.reshape(1,-1) == values.reshape(-1,1)
    cont_matrix = (first_rater[np.newaxis,::] * second_rater[:,np.newaxis,:]).sum(2) / num_samples
   
    first_marginals = cont_matrix.sum(0)
    second_marginals = cont_matrix.sum(1)
    PO = np.sum(np.diag(cont_matrix))
    PE = np.sum(first_marginals * second_marginals)
    kappa = (PO - PE) / (1- PE)

    M  = cont_matrix * (first_marginals[np.newaxis] + second_marginals[:,np.newaxis])**2
    M_sum_diag = np.sum(np.diag(M))
    M_sum_nondiag = M.sum() - M_sum_diag

    Var =  (PO*(1-PE)**2 + (1-PO)**2 *M_sum_nondiag -  (1-PO)**2 * M_sum_diag - (PO*PE - 2*PE +PO)**2 ) / (1-PE)**4 / num_samples
    delta = np.sqrt(Var) * z
    return kappa, delta


def load_agreement_keypoints(agreement_yml_path):
    """
    Loads keypoints for agreement studies
    
    Parameters
    -----------
    agreement_yml_path : str
        path to yml with lists for agreement study
    
    Returns
    -------
    keypoints : dict of endoanalysis.targets.KeypointsTruthBatch
        loaded keypoints
        
    Note
    ----
    The agreement yml should have the following structure:
    
    expert1:
      images_list: relative/path/to/images/list/for/exprert1/images.txt
      labels_list: relative/path/to/images/list/for/exprert1/labels.txt
    expert2:
      images_list: relative/path/to/images/list/for/exprert2/images.txt
      labels_list: relative/path/to/images/list/for/exprert2/labels.txt
      
    and so on.
      
    """
    with open(agreement_yml_path, "r") as file:
        lists = yaml.safe_load(file)
        
    datasets = {}
    keypoints = {}
    datasets_lens = {}
    for expert_name in lists.keys():

        images_list = lists[expert_name]["images_list"]
        labels_list = lists[expert_name]["labels_list"]
        images_list = os.path.normpath(os.path.join(os.path.dirname(agreement_yml_path), images_list))
        labels_list = os.path.normpath(os.path.join(os.path.dirname(agreement_yml_path), labels_list))

        dataset = PointsDataset(
            images_list,
            labels_list
        )
        datasets_lens[expert_name] = len(dataset)
        datasets[expert_name] = dataset
        
        keypoints[expert_name] = dataset.collate([dataset[x] for x in range(len(dataset))])["keypoints"]
        
        datasets_lens_list = list(datasets_lens.values())
        
    if len(np.unique(datasets_lens_list)) != 1:
        raise Exception("Some annotators have different numbe of images than the others")
    else:
        num_images = datasets_lens_list[0]
        
    return keypoints


def plot_agreement_matrix(kappas, experts, precision=2, fig= None, ax=None):
    
    if (ax is None and fig is not None) or (ax is not None and fig is None):
        raise Exception("Fig and ax paramterers must be None or not None simultaneously")
        
    kappas = np.copy(kappas)
    number_format = "1.%if"%precision
    mask = np.zeros_like(kappas)
    mask[np.triu_indices_from(kappas,k=1)] = True
    with sns.axes_style("white"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            kappas, 
            mask=mask,
            vmin=0.,  
            vmax=1,
            annot=True, 
            fmt=number_format,
            square=True,
            cbar=False,
            cmap="coolwarm",
            ax=ax
        )
        ax.set_xticklabels(experts)
        ax.set_yticklabels(experts)
        ax.hlines([3], 0., 3., color="black", lw=3.)
        ax.vlines([3], 3., 7., color="black", lw=3.)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45 )
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=45 )
    
    return fig, ax

import numpy as np
import seaborn as sns

def plot_confidence_intevals_matrix(kappas, deltas, experts, precision=2, fig= None, ax=None):
    
    if (ax is None and fig is not None) or (ax is not None and fig is None):
        raise Exception("Fig and ax paramterers must be None or not None simultaneously")
        
    kappas = np.copy(kappas)
    mask = np.zeros_like(kappas)
    mask[np.triu_indices_from(kappas,k=1)] = True
    kappas_low = kappas - deltas
    kappas_high = kappas + deltas
    kappas_low = np.round(kappas - deltas, precision)
    kappas_high = np.round(kappas + deltas, precision)
    annotation = np.core.defchararray.add(kappas_low.astype(str), " - ") 
    annotation = np.core.defchararray.add(annotation, kappas_high.astype(str))
    annotation[np.diag_indices_from(annotation)] = "-"
    with sns.axes_style("white"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            kappas, 
            mask=mask,
            vmin=0.,  
            vmax=1,
            annot=annotation, 
            fmt="",
            square=True,
            cbar=False,
            cmap="coolwarm",
            linewidths=1,
            annot_kws={"fontsize":9},
            ax=ax
        )
        ax.set_xticklabels(experts)
        ax.set_yticklabels(experts)
        ax.hlines([3], 0., 3., color="black", lw=3.)
        ax.vlines([3], 3., 7., color="black", lw=3.)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45 )
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=45 )
    
    return fig, ax


def plot_agreement_matrices(kappas, studies, experts, precision=2, fig_size=(16, 4)):
    fig, axs = plt.subplots(1,len(studies), figsize=fig_size)
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.1)
    for ax, study_name in zip(axs, studies):
        plot_agreement_matrix(kappas[study_name], experts, precision=precision, fig=fig, ax=ax)
        ax.set_title(study_name, y=-0.25)
    fig.colorbar(cm.ScalarMappable(norm=None, cmap="coolwarm"), ax=ax)
    return fig, ax

def plot_confidence_intervals_matrices(kappas, deltas, studies, experts, precision=2, fig_size=(16, 4)):
    fig, axs = plt.subplots(1,len(studies), figsize=fig_size)
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.1)
    for ax, study_name in zip(axs, studies):
        plot_confidence_intevals_matrix(kappas[study_name], deltas[study_name], experts, precision=precision, fig=fig, ax=ax)
        ax.set_title(study_name, y=-0.25)
    fig.colorbar(cm.ScalarMappable(norm=None, cmap="coolwarm"), ax=ax)
    return fig, ax

def get_num_images(keypoints, experts):
    num_images = keypoints[experts[0]].num_images()
    for expert in experts[:1]:
        num_images_check = keypoints[expert].num_images()
        if num_images != num_images_check:
            raise Exception("Num images for experts %s and %s are different"%(experts[0], expert))
    return num_images

def get_keypoints_nums(keypoints, experts_mapping):
    experts = list(experts_mapping.keys())
    
    num_images = get_num_images(keypoints, experts)
            

    keypoints_nums = np.zeros((len(experts), num_images))
    
    for expert, expert_i in experts_mapping.items():
        keypoints_num_expert = [len(keypoints[expert].from_image(image_i)) for image_i in range(num_images)]
        keypoints_nums[expert_i] = np.array(keypoints_num_expert)
        
    return keypoints_nums

def compute_icc(keypoints_nums,  experts):
    num_images = keypoints_nums.shape[1]
    df_dict = {
        "rater": np.arange(len(experts)).repeat(num_images),
        "image":  np.hstack([np.arange(num_images)]* len(experts)),
        "nums": keypoints_nums.flatten(order="C")
    }
    df = pd.DataFrame(df_dict)
    icc = pg.intraclass_corr(data=df, targets='image', raters='rater',ratings='nums').round(3)
    return icc

def plot_numbers(keypoints_nums, experts,experts_mapping, width=0.11, fig=None, ax=None):
  
    if (ax is None and fig is not None) or (ax is not None and fig is None):
        raise Exception("Fig and ax paramterers must be None or not None simultaneously")
    
    num_images = keypoints_nums.shape[1]
    x = np.arange(num_images)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(20,5))
 
    num_experts = len(experts)
    handles = []
    for expert, expert_i in experts_mapping.items():
        ax.set_xticks(np.arange(num_images))
        handle = ax.bar(
            x - (expert_i -  num_experts/2.)*width , 
            keypoints_nums[expert_i], 
            width=width , 
            align='center', 
            edgecolor='black', 
            zorder=1,
            label=expert
        )
        handles.append(handle)
    ax.grid(b=True,axis="y", zorder=2)
    ax.set_xlabel("Images")
    ax.set_ylabel("Nuclei counts")
    return fig, ax, handles
    
        
def ptg_agreement(kappas, experts, experts_mapping):
    kappas_to_mean = []
    for expert_1, expert_2 in itertools.combinations(experts, 2):
        if "ptg" in expert_1 and "ptg" in expert_2:
            kappas_to_mean.append(kappas[experts_mapping[expert_1], experts_mapping[expert_2]])
    return np.mean(kappas_to_mean)

def stud_agreement(kappas, experts, experts_mapping):
    kappas_to_mean = []
    for expert_1, expert_2 in itertools.combinations(experts, 2):
        if "stud" in expert_1 and "stud" in expert_2:
            kappas_to_mean.append(kappas[experts_mapping[expert_1], experts_mapping[expert_2]])
    return np.mean(kappas_to_mean)

def ptg_stud_agreement(kappas, experts, experts_mapping):
    kappas_to_mean = []
    for expert_1, expert_2 in itertools.combinations(experts, 2):
        if ("ptg" in expert_1 and "stud" in expert_2) or ("stud" in expert_1 and "ptg" in expert_2):
            kappas_to_mean.append(kappas[experts_mapping[expert_1], experts_mapping[expert_2]])
    return np.mean(kappas_to_mean)