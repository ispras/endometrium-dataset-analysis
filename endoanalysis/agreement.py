import itertools
import copy
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score


class SimilarityMeasure:
    """
    Base class for similarity measures
    """

    def measure(self, object1, object2):
        """
        Returns the measure value between two objects
        """
        return 0

    def matrix(self, container1, container2):
        """
        Returns the matrix of measure values between two sets of objects
        Sometimes can be implemented in a faster way than making couplewise measurements
        """
        matrix = np.zeros((len(container1), len(container2)))
        for i, object1 in enumerate(container1):
            for j, object2 in enumerate(container2):
                matrix[i, j] = self.measure(object1, object2)
        return matrix


class Minkovsky2DSimilarity(SimilarityMeasure):
    def __init__(self, p=2, scale=1.0):
        self.p = p
        self.scale = scale

    def measure(self, point1, point2):
        diff = np.abs(point1 - point2)
        power = np.power(diff, self.p)
        distance = np.power(power.sum(), 1 / self.p) / self.scale
        return distance

    def matrix(self, points1, points2):

        coords1 = np.vstack([points1.x_coords(), points1.y_coords()])
        coords2 = np.vstack([points2.x_coords(), points2.y_coords()])
     
        diffs = np.abs(coords1[:, :, np.newaxis] - coords2[:, np.newaxis, :])

        powers = np.power(diffs, self.p)
        matrix = np.power(powers.sum(axis=0), 1 / self.p) / self.scale

        return matrix


class OKSimilarity(Minkovsky2DSimilarity):
    """
    Object keypoints similarity
    """

    def __init__(self, p=2, scale=1.0):
        super().__init__(p, scale)

    def _exp_square(self, arr):
        return np.exp(-np.power(arr, 2) / 2.0)

    def measure(self, point1, point2):
        distance = super().measure(point1, point2)
        return self._exp_square(distance)

    def matrix(self, points1, points2):
        distance_matrix = distance = super().matrix(points1, points2)
        return self._exp_square(distance)
    
    
    

def get_batch_relaibility_matrix(
    targets_from_batch1, 
    targets_from_batch2, 
    similarity,
    indexes_to_consider 
):
    '''
    Composes relaibility matrix for two batches.
    
    Parameters
    ----------
    targets_from_batch1 : endoanalysis.targets.TargetsBatchArray like
        batch of targets corresponding to a set of images from the first rater in pair. 
        Must have have the method targets_from_batch1.from_image(), 
        returning the container of targets corrsponding to a given image.
        The container must be compatible with similarity.
    targets_from_batch2 :  endoanalysis.targets.TargetsBatchArray  like
        the same as targets_from_batch1, but for the second rater.
    similarity : endoanalysis.agreement.SimilarityMeasure
        similarity measure, must be compatible with targets_from_batch1 and targets_from_batch2
    indexes_to_consider : iterable
        an iterable of image indexes which sould be taken into concideration. 
        All indexes must be present in both  targets_from_batch1 an targets_from_batch2
        
    Returns
    -------
    relaibility_matrix : ndarray
        relaibility matrix. The spae is (2, num_matched), where num_matched
        is the number of targets which were sucsefully matched. This number
        could not be gereater than the maximum total number of targets in
        batch1 or batch2
    '''
    
    rel_matrices = []
    for image_i in indexes_to_consider:
        targets1 = targets_from_batch1.from_image(image_i)
        targets2 = targets_from_batch2.from_image(image_i)

        sim_matrix = similarity.matrix(targets1, targets2)

        row_ids, col_ids = linear_sum_assignment(sim_matrix, maximize=True)
        rel_matrices.append(np.vstack([targets1.classes()[row_ids], targets2.classes()[col_ids]]))
    relaibility_matrix = np.hstack(rel_matrices)
    
    return relaibility_matrix

def compute_kappas(
    targets_containers,
    similarity,
    raters_list,
    indexes_to_consider,
):
    '''
    Compute Cohen's kappa for all raters from raters_list.  
    
    Parameters
    ---------
    targets_containers : dict of endoanalysis.targets.TargetsBatchArray
        a dict with batches of targets. The keys are the raters ids.
        Targets batches should be compatible with similarity
    similarity : endoanalysis.agreement.SimilarityMeasure
        similarity measure, must be compatible with targets_from_batch1 and targets_from_batch2
    raters_list : iterable
        an iterable of raters to consider. All listed raters must be in targets_containers keys
    indexes_to_consider : iterable
        an iterable of image indexes which sould be taken into concideration. 
        All indexes must be present in all targets_containers
    
    Returns
    -------
    kappas : list of float
        list of kappas for all raters pairs
    '''

    
    kappas = []
    for expert_1, expert_2 in itertools.combinations(raters_list, 2):
        relaibility_matrix = get_batch_relaibility_matrix(
            targets_containers[expert_1],
            targets_containers[expert_2],
            similarity,
            indexes_to_consider
        )
        kappa = cohen_kappa_score(relaibility_matrix[0], relaibility_matrix[1])
        kappas.append(kappa)
    
    return kappas
        
    
    
# def compose_samples_as_pairs(objects_1, objects_2, similarity, sim_thresh):
    
#     '''
#     Finds all pairs of objects(keypoints, bboxes, etc.) taken from two groups which are above the specified similarity threshold. 
    
#     Parameters
#     ----------
#     objects_1 : ndarray
#         first group of the objects. Must be compatible with similiarty.
#     objects_2 : ndarray
#         second group of the objects. Must be compatible with similiarty.
#     similarity : endoanalysis.agreement.SimilarityMeasure
#         a measure of simelarity between two objects
#     sim_thersh : float
#         the threshold for similarity value defining whether two objects are
#         corresponding to the one entity or not
        
#     Returns
#     -------
#     pairs : ndarray
#         array with the indices of the pairs. The shape is (num_pairs, 3). 
#         Each row corresponds to a pair and has the following values(signatures):
#         (similiarty value, index in objects_1, index in objects_2).
#         The unpaired objects are also put here and has the signature (-1, index, -1) for the objects from objects_1 and
#         (-1, -1, index) for the objects from objects_2
#         The order of pairs is the following: first come the paired objects, than the unpaired objects from objects_1
#         and than the unpaired objects from objects_2
        
#     See also
#     --------
#     endoanalysis.agreement.SimilarityMeasure 
        
#     '''
    
#     sim_matrix = similarity.matrix(
#             objects_1,
#             objects_2
#         )
    
#     num_rows, num_cols = sim_matrix.shape
#     pairs = []
#     unpaired_first = set(range(num_rows))
#     unpaired_second = set(range(num_cols))
    
#     for row_i in range(num_rows):
#         for col_j in range(num_cols):
#             if sim_matrix[row_i, col_j] >= sim_thresh:
#                 pairs.append((sim_matrix[row_i, col_j]*1000, row_i, col_j))
#                 if row_i in unpaired_first:
#                     unpaired_first.remove(row_i)
#                 if col_j in unpaired_second:
#                     unpaired_second.remove(col_j)
          
#     if len(pairs):
#         pairs = np.array(pairs, dtype=int)
#         pairs_sorted = pairs[np.argsort(pairs[:,0])][::-1]
        
#     else:
#         pairs = np.empty((0,3))
    
#     unpaired_first = np.array(list(unpaired_first), dtype=int)
#     unpaired_second = np.array(list(unpaired_second), dtype=int)
#     unpaired_first = np.vstack([np.ones_like(unpaired_first) * (-1), unpaired_first,  np.ones_like(unpaired_first) * (-1)]).transpose()
#     unpaired_second = np.vstack([np.ones_like(unpaired_second) * (-1), np.ones_like(unpaired_second) * (-1), unpaired_second ]).transpose()
   
#     pairs = np.vstack([pairs, unpaired_first, unpaired_second])
 
#     return pairs


# def glue_samples(sample1, sample2):
#     '''
#     Checks whether two samples can be glued togeter and glues them if it is the case
    
#     Parameters
#     ----------
#     sample1: ndarray
#         array with the indices of the first sample. The shape is (num_experts,).
#         The -1 value corresponds to a no opinion
#     sample2: ndarray
#         array with the indices of the second sample. The shape is (num_experts,)
        
#     Returns
#     -------
#     is_compatible : bool
#         the flag indicating that the samples can be glued together
#     new_sample : list
#         the reult of glueing procedure. If is_compatible is False,
#         an empty list is returned
        
#     Note
#     ----
#     Two samples can be glued together if two conditions are satisfied:
#     1) There are at least two experts who marked the sample with similar indices
#     2) There are no experts who marked the sample with different indices
    
#     If the expert didn't marked the sample the index is -1), his field is ignored
#     '''
    
#     new_sample = []
    
#     for index1, index2 in zip(sample1, sample2):
#         if index1 == -1:
#             index1 = index2
#         if index2 == -1:
#             index2 = index1
#         if index1 != index2:
#             return False, []
#         new_sample.append(index1)
        
#     return True, new_sample

# def merge_samples(samples_image):
#     '''
#     Merges the samples for a given image
    
#     Parameters
#     ----------
#     samples_image : ndaarray
#         samples to merge. Should have the shape (num_pairs, num_experts). 
#         Assumed to be sorted by similarities.
        
#     Returns
#     -------
#     samples_merged : ndarray
#         merged samples. The shape is the same as samples_image.shape
        
#     '''
#     samples_merged = copy.deepcopy(samples_image)
#     num_experts = samples_merged.shape[1]
  
#     for i, sample in enumerate(samples_merged[1:], start=1):
#         for j, base_sample in enumerate(samples_merged[0:i]):
#             compat, new_sample = glue_samples(sample, base_sample)
#             if compat:
#                 samples_merged[j] = new_sample
#                 samples_merged[i] = np.array([-1] * num_experts)

#     return samples_merged     


# def compose_relaibility_matrix(objects_image, objects_classes, experts_list, similarity, similarity_thresh):
#     '''
#     Composes relaibility matrix for the objects (keypoints, bboxes, etc.) for a given image.
    
#     Parameters
#     ----------
#     objects_image : dict of ndarray
#         a dictionary with the arrays of objects. The keys are the experts names.
#         ndarrays of objects must be compatible with similarity.
#     objects_classes : dict of ndarray
#         a dictionary with the arrays of objects classes. The keys are the experts names.
#         ndarrays has the shape (num_keypoints,) and the entities are class labels
#     experts_list : list of str
#         a list of experts names to conseder. 
#         If there is a key in objects_image which is not in experts_list, it will be ignored.
#     similarity : endoanalysis.agreement.SimilarityMeasure
#         a measure of simelarity between two objects.
#     similarity_thresh : float
#         the threshold for similarity value defining whether two objects are
#         corresponding to the one entity or not.
        
#     Returns
#     -------
#     relaibility_matrix : ndarray
#         an array with the shape (num_objects, num_experts).
#         The element at (i,j) position encodes the class, which jth expert gave to ith entity.
#         If the entity is not labeled by the expert, the -1 is assigned 
        
#     Note
#     ----
#     The algorithm has the following steps:
    
#     1) For each pair of expert compose the pairs with compose_samples_as_pairs
#     2) Add additional columns filled with -1 to the resulting arrays to make their shape (num_pairs, num_experts)
#     3) Sort the pairs according to similarity vvalues (first column)
#     4) Cut off similarity values
#     5) Merge the pairs with merge_samples
#     6) Erase the samles which occasianlly got (-1, -1, ... -1) signatures after the merging
#     7) Put the the objects claasses assigned by the experts instead of indices
    
#     See also
#     --------
#     endoanalysis.agreement.SimilarityMeasure
#     endoanalysis.agreement.compose_samples_as_pairs
#     endoanalysis.agreement.merge_samples
#     '''

#     samples_image = []

#     for expert_i, expert_j in itertools.combinations(range(len(experts_list)), 2):


#         samples_as_pairs  = compose_samples_as_pairs(
#             objects_image[experts_list[expert_i]], 
#             objects_image[experts_list[expert_j]], 
#             similarity, 
#             sim_thresh=similarity_thresh
#         )
        
#         num_samples = len(samples_as_pairs)

#         # inserting additional columns for the other experts
#         samples_as_pairs = [samples_as_pairs[:,0]] +\
#             [np.ones(num_samples) * (-1)] * expert_i +\
#             [samples_as_pairs[:,1]] +\
#             [np.ones(num_samples) * (-1)] * (expert_j - expert_i - 1 ) +\
#             [[samples_as_pairs[:,2]]] +\
#             [np.ones(num_samples) * (-1)] * (len(experts_list) - expert_j - 1)

#         samples_as_pairs = np.vstack(samples_as_pairs).astype(int).transpose()
#         samples_image.append(samples_as_pairs)

#     samples_image = np.vstack(samples_image).astype(int)
    

    
#     #sorting with the similarities, so the pairs with the largest similarity goes to the beginning
#     samples_image = samples_image[np.argsort(samples_image[:,0])][::-1]
    
#     #cutting off similartity values
#     samples_image = samples_image[:,1:]
    
#     #merging the samples
#     samples_image = merge_samples(samples_image)
    

      
#     #erasing the samples with (-1, -1, ..., -1) signatures     
#     samples_image = samples_image[np.where(np.prod(samples_image==-1, axis=1) == False)]     
    
#     #computing relaibility matrix based on classes
#     relaibility_matrix = np.zeros_like(samples_image)
#     for expert_i, expert in enumerate(experts_list):
#         classes = objects_classes[expert]
#         samples_ids = samples_image[:,expert_i]
#         relaibility_matrix[:, expert_i] = np.vectorize(lambda x: classes[x] if x>=0 else x)(samples_ids)
        
#     return relaibility_matrix


# def compose_batch_relaibility_matrix(
#     objects_batches,
#     experts_list, 
#     similarity, 
#     sim_thresh, 
#     images_indexes, 
#     missings_as_classes = True, 
#     class_agnostic = False
# ):
#     '''
#     Composes relaibility matrix for a batch of objects((keypoints, bboxes, etc.))
    
#     Parameters
#     ----------
#     objects_batches : dict of endoanalysis.keypoints.KeypointsArray
#         a batch of keypoints. The keys are experts names
#     experts_list : list of str
#         a list of experts names to conseder. 
#         If there is a key in objects_image which is not in experts_list, it will be ignored.
#     similarity : endoanalysis.agreement.SimilarityMeasure
#         a measure of simelarity between two objects.
#     similarity_thresh : float
#         the threshold for similarity value defining whether two objects are
#         corresponding to the one entity or not.
#     missings_as_classes : bool
#         whether to treat the objects which are not found by some experts as separate classes
#     class_agnostic : bool
#         ignore the class labels (all classes are set to one). 
#         Makes sense only if missings_as_classes is True
    
#     Returns
#     -------
#     relaibility_matrix : ndarray
#         an array with the shape (num_objects, num_experts).
#         The element at (i,j) position encodes the class, which jth expert gave to ith entity.
#         If the entity is not labeled by the expert, the nan is assigned. If missings_as_classes is True,
#         there are -1s except for nans
    
#     '''
    
    
#     reliability_matrix = []
#     print("Composing relaibility matrix")
#     with tqdm(total = len(images_indexes)) as pbar:
#         for image_i in images_indexes:

#             rel_matrix_image = compose_relaibility_matrix( 
#                 {expert : objects_batches[expert].from_image(image_i) for expert in experts_list},
#                 {expert : objects_batches[expert].from_image(image_i).classes() for expert in experts_list},
#                 experts_list,
#                 similarity,
#                 similarity_thresh=sim_thresh
#             )
#             reliability_matrix.append(rel_matrix_image)
#             pbar.update()

#     reliability_matrix = np.vstack(reliability_matrix).transpose()

    
#     if class_agnostic:
#         reliability_matrix[np.where(reliability_matrix!=-1)] = 1
    
#     if not missings_as_classes:
#         reliability_matrix = reliability_matrix.astype(float)
#         reliability_matrix[np.where(reliability_matrix==-1)] = np.nan
#     return reliability_matrix