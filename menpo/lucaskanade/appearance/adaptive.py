from scipy.linalg import norm
import numpy as np
from menpo.lucaskanade.appearance.base import AppearanceLucasKanade


class AdaptiveForwardAdditive(AppearanceLucasKanade):

    type = 'AdaFA'

    def _align(self, lk_fitting, max_iters=20, project=True):
        # Initial error > eps
        error = self.eps + 1
        image = lk_fitting.image
        lk_fitting.weights = []
        n_iters = 0

        # Initial appearance weights
        if project:
            # Obtained weights by projection
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)
            weights = self.appearance_model.project(IWxp)
            # Reset template
            self.template = self.appearance_model.instance(weights)
        else:
            # Set all weights to 0 (yielding the mean)
            weights = np.zeros(self.appearance_model.n_active_components)

        lk_fitting.weights.append(weights)

        # Forward Additive Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current parameters
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)

            # Compute warp Jacobian
            dW_dp = self.transform.jacobian(
                self.template.mask.true_indices)

            # Compute steepest descent images, VI_dW_dp
            J_aux = self.residual.steepest_descent_images(
                image, dW_dp, forward=(self.template, self.transform,
                                       self._interpolator))

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model.project_out_vectors(J_aux.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            parameters = self.transform.as_vector() + delta_p
            self.transform.from_vector_inplace(parameters)
            lk_fitting.parameters.append(parameters)

            # Update appearance weights
            error_img = self.template.from_vector(
                self.residual._error_img - np.dot(J_aux, delta_p))
            weights -= self.appearance_model.project(error_img)
            self.template = self.appearance_model.instance(weights)
            lk_fitting.weights.append(weights)

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        lk_fitting.fitted = True
        return lk_fitting


class AdaptiveForwardCompositional(AppearanceLucasKanade):

    type = 'AdaFC'

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.transform.jacobian(
            self.template.mask.true_indices)

    def _align(self, lk_fitting, max_iters=20, project=True):
        # Initial error > eps
        error = self.eps + 1
        image = lk_fitting.image
        lk_fitting.weights = []
        n_iters = 0

        # Initial appearance weights
        if project:
            # Obtained weights by projection
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)
            weights = self.appearance_model.project(IWxp)
            # Reset template
            self.template = self.appearance_model.instance(weights)
        else:
            # Set all weights to 0 (yielding the mean)
            weights = np.zeros(self.appearance_model.n_active_components)

        lk_fitting.weights.append(weights)

        # Forward Additive Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current parameters
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)

            # Compute steepest descent images, VI_dW_dp
            J_aux = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model.project_out_vectors(J_aux.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            self.transform.compose_after_from_vector_inplace(delta_p)
            lk_fitting.parameters.append(self.transform.as_vector())

            # Update appearance weights
            error_img = self.template.from_vector(
                self.residual._error_img - np.dot(J_aux, delta_p))
            weights -= self.appearance_model.project(error_img)
            self.template = self.appearance_model.instance(weights)
            lk_fitting.weights.append(weights)

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        lk_fitting.fitted = True
        return lk_fitting


class AdaptiveInverseCompositional(AppearanceLucasKanade):

    type = 'AdaIC'

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.transform.jacobian(
            self.template.mask.true_indices)

    def _align(self, lk_fitting, max_iters=20, project=True):
        # Initial error > eps
        error = self.eps + 1
        image = lk_fitting.image
        lk_fitting.weights = []
        n_iters = 0

        # Initial appearance weights
        if project:
            # Obtained weights by projection
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)
            weights = self.appearance_model.project(IWxp)
            # Reset template
            self.template = self.appearance_model.instance(weights)
        else:
            # Set all weights to 0 (yielding the mean)
            weights = np.zeros(self.appearance_model.n_active_components)

        lk_fitting.weights.append(weights)

        # Baker-Matthews, Inverse Compositional Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current parameters
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)

            # Compute steepest descent images, VT_dW_dp
            J_aux = self.residual.steepest_descent_images(self.template,
                                                          self._dW_dp)

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model.project_out_vectors(J_aux.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Request the pesudoinverse vector from the transform
            inv_delta_p = self.transform.pseudoinverse_vector(delta_p)

            # Update warp parameters
            self.transform.compose_after_from_vector_inplace(inv_delta_p)
            lk_fitting.parameters.append(self.transform.as_vector())

            # Update appearance parameters
            error_img = self.template.from_vector(
                self.residual._error_img - np.dot(J_aux, delta_p))
            weights -= self.appearance_model.project(error_img)
            self.template = self.appearance_model.instance(weights)
            lk_fitting.weights.append(weights)

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        lk_fitting.fitted = True
        return lk_fitting
