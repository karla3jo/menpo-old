# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
from numpy.linalg.linalg import LinAlgError

import scipy.io
import numpy as np
from pybug.lucaskanade.vector_utils import Spherical
from pybug.transform import SimilarityTransform, Translation
from pybug.lucaskanade.image import ImageInverseCompositional
from pybug.image import MaskedNDImage, BooleanNDImage
import matplotlib.pyplot as plt
from pybug.shape import PointCloud
from pybug.io import auto_import
import os

# <codecell>

def rms_point_error(original_box, transformed_box):
    return np.sqrt(np.mean((original_box.points - transformed_box.points) ** 2))

# <codecell>

def build_normal_image(image):
    n_image = MaskedNDImage.blank(image.shape, mask=image.mask, n_channels=3)
    n_image.from_vector_inplace(image.mesh.vertex_normals.ravel())
    return n_image

def build_spher_image(image):
    n_image = MaskedNDImage.blank(image.shape, mask=image.mask, n_channels=4)
    normals = image.mesh.vertex_normals
    spher = Spherical().logmap(normals)
    n_image.from_vector_inplace(spher.ravel())
    return n_image

# <codecell>

from pybug.lucaskanade import LSIntensity
from pybug.lucaskanade.residual import LSIPNormalise, LSSpherNormalise, NormalInnerProductCorrelation

def get_residual(option):
    if option == 'DEPTH':
        return (lambda x: x.as_depth_image(), LSIntensity())
    elif option == 'NORMAL':
        return (build_normal_image, LSIntensity())
    elif option == 'IP_FS':
        return (build_normal_image, LSIPNormalise())
    elif option == 'SPHER_FS':
        return (build_spher_image, LSSpherNormalise())
    elif option == 'PQ_ECC':
        return (lambda x: x.as_depth_image(), NormalInnerProductCorrelation())
    else:
        raise ValueError('Unknown algorithm option')

# <codecell>

def my_test_affine(tdata, pt_offsets, alg_list, n_iters, n_freq_tests, spatial_sigma, verbose):
    results = {}
    
    bounds_height = tdata.bounds_height
    bounds_width = tdata.bounds_width
    bounds = tdata.bounds
    img = tdata.img
    template = tdata.template
    
    # The box that the face lives in (not offset to correct position)
    original_box = PointCloud(np.array([[0,             0],
                                        [bounds_height, 0],
                                        [bounds_height, bounds_width],
                                        [0,             bounds_width]]))
    
    # The initial params for the algorithm (rough initialisaton)
    initial_params = np.array([0, 0, bounds[0][0] + 0.5, bounds[0][1] + 0.5])
    # A transform that applies the offset
    offset_transform = Translation(np.array([bounds[0][0], bounds[0][1]]))
    
    # Run
    for offset_idx in xrange(n_freq_tests):
        # TODO: use pt_offset
        # Perturb the original box by some small amount
        target_box = PointCloud(original_box.points + np.random.uniform(-1, 1, (4, 2)) * spatial_sigma)
        # Snap the affine perturbation back to a similarity transformation
        target_transform = SimilarityTransform.align(original_box, target_box)
        # Create the target box (not offset)
        target_transform.apply_inplace(target_box)

        # Warp original image to get test "template" image
        template_mask = BooleanNDImage.blank((bounds_height, bounds_width))
        # Compose with the offset so that it's at the correct position on the face
        target = template.warp_to(template_mask, target_transform.compose_after(offset_transform))

        if verbose:
            print 'Initial RMS: {0}'.format(rms_point_error(original_box, target_box))
        
        # Run each algorithm
        for i, option in enumerate(alg_list):
            # Allow the passing of arguments in to the instantiated class
            preprocess_func, metric = get_residual(option)
            
            processed_target = preprocess_func(target)
            processed_img = preprocess_func(img)
            try:
                iic = ImageInverseCompositional(processed_target, metric, SimilarityTransform(np.eye(3)), interpolator='c')
                final_transform = iic.align(processed_img, initial_params, max_iters=n_iters)
                # Create the estimated box
                estimated_box = final_transform.apply(original_box)
                # Make sure the original box is at the correct offset
                rms_pt_error = rms_point_error(offset_transform.apply(original_box), estimated_box)
            except LinAlgError:
                print '{} Diverged'.format(option)
                rms_pt_error = 1000.0

            if verbose:
                print '{0}: {1}'.format(option, rms_pt_error)

            if not option in results:
                results[option] = []
            measure_results = results[option]
            measure_results.append(rms_pt_error)
            results[option] = measure_results

    return results

# <codecell>

# Load datasets
np.set_printoptions(suppress=True, precision=3, linewidth=600)
bosphorus_path = '/vol/hci2/Databases/video/Bosphorus/BosphorusDB/'
subject_list = ['bs000', 'bs001', 'bs003']
neutral_id = '_N_N_0'
expression_list = ['_O_EYE_0', '_O_MOUTH_0', '_O_GLASSES_0', '_O_HAIR_0']

num_of_subjs = len(subject_list)
num_of_imgs_per_subj = len(expression_list)

# Set up experiment variables
verbose = False
n_iters = 30                     # Number of gradient descent iterations
n_freq_tests = 100               # Number of frequency of convergence tests
max_spatial_error = 3.0          # Max location error for deciding convergence
all_spc_sig = np.arange(1, 11)   # All spatial sigmas (1,10)

alg_list = ['DEPTH', 'NORMAL', 'IP_FS', 'SPHER_FS']

results = np.zeros([num_of_subjs, num_of_imgs_per_subj, len(all_spc_sig), len(alg_list)])

# <codecell>

# Run experiment for each subject
for subj in xrange(num_of_subjs):
    subject_id = subject_list[subj]
    print 'Subject {} of ({}/{})'.format(subject_id, subj + 1, num_of_subjs)
    template = auto_import(os.path.join(bosphorus_path, subject_id, '{}{}.bnt'.format(subject_id, neutral_id)), verbose=False)[0]
                       
    for subj_img in xrange(num_of_imgs_per_subj):
        expression_id = expression_list[subj_img]
        img = auto_import(os.path.join(bosphorus_path, subject_id, '{}{}.bnt'.format(subject_id, expression_id)), verbose=False)[0]

        template_labels = set(template.landmarks['LM2'].labels)
        img_labels = set(img.landmarks['LM2'].labels)
        matching_labels = template_labels.intersection(img_labels)
        
        template_points = template.landmarks['LM2'].with_labels(matching_labels).lms
        img_points = img.landmarks['LM2'].with_labels(matching_labels).lms
        
        transform = SimilarityTransform.align(template_points, img_points)
        
        img = img.warp_to(BooleanNDImage.blank(template.shape), transform, warp_mask=True, warp_landmarks=True)
        
        template.landmarks['face'] = template.landmarks['LM2']#.with_labels(['middle_left_eyebrow',
                                                              #              'middle_right_eyebrow',
                                                              #              'lower_lip_outer_middle']).lms
        
        bounds = template.landmarks['face'].lms.bounds()
        bounds_height, bounds_width = (bounds[1][1] - bounds[0][1], bounds[1][0] - bounds[0][0])
        
        tdata = lambda x: 0
        tdata.template = template
        tdata.img = img
        tdata.bounds = bounds
        tdata.bounds_height = bounds_height
        tdata.bounds_width = bounds_width
        
        # Run tests
        for sigma_ind, current_sigma in enumerate(all_spc_sig):
            # TODO: generate pt_offset
            res = my_test_affine(tdata, [], alg_list, n_iters, n_freq_tests, current_sigma, verbose)

            for measure_ind, option in enumerate(alg_list):
                measure_results = res[option]
                # Get whether or not it converges
                n_converge = len(filter(lambda error: error < max_spatial_error, measure_results))
                results[subj, subj_img, sigma_ind, measure_ind] = n_converge


# Save out results just in case
scipy.io.savemat('results.mat', {'results': results})

# <codecell>

# Plot results
mean_results = np.mean(np.mean(results, 1), 0) / float(n_freq_tests)


line_styles = ['k--D', 'y:^', 'r:*', 'g:^', 'b-s']
lines = []
for i in xrange(mean_results.shape[1]):
    lines.append(all_spc_sig)
    lines.append(mean_results[:, i])
    lines.append(line_styles[i])

p = plt.plot(*lines)

legend_labels = alg_list
plt.yticks(np.linspace(0, 1, 11))
plt.xticks(all_spc_sig)
plt.ylabel('Frequency of Convergence')
plt.xlabel('Point Standard Deviation')
plt.legend(p, legend_labels)
plt.title('Bosphorus')

plt.savefig('result.png')

# <codecell>


