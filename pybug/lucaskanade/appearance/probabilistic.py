from scipy.linalg import norm
import numpy as np
from pybug.lucaskanade.appearance.base import AppearanceLucasKanade


class ProbabilisticForwardAdditive(AppearanceLucasKanade):

    def _align(self, max_iters=50, alpha=1):
        # Initial error > eps
        error = self.eps + 1

        # Set relative weighting between distances
        beta = 2 - alpha

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.transform,
                                      interpolator=self._interpolator)

            # Compute warp Jacobian
            dW_dp = self.transform.jacobian(
                self.template.mask.true_indices)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(
                self.image, dW_dp, forward=(self.template,
                                            self.transform,
                                            self._interpolator))

            # Project out appearance model from VI_dW_dp
            self._J = (alpha *
                       self.appearance_model.distance_to_subspace_vector(J.T) +
                       beta *
                       self.appearance_model.project_whitened_vector(J.T)).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            params = self.transform.as_vector() + delta_p
            self.transform.from_vector_inplace(params)
            self.parameters.append(params)

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.transform


class ProbabilisticForwardCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.transform.jacobian(
            self.template.mask.true_indices)

        pass

    def _align(self, max_iters=50, alpha=1):
        # Initial error > eps
        error = self.eps + 1

        # set relative weighting between distances
        beta = 2 - alpha

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.transform,
                                      interpolator=self._interpolator)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Project out appearance model from VI_dW_dp
            self._J = (alpha *
                       self.appearance_model.distance_to_subspace_vector(J.T) +
                       beta *
                       self.appearance_model.project_whitened_vector(J.T)).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            self.transform.compose_after_from_vector_inplace(delta_p)
            self.parameters.append(self.transform.as_vector())

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.transform


class ProbabilisticInverseCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        dW_dp = self.transform.jacobian(
            self.template.mask.true_indices)

        # Compute steepest descent images, VT_dW_dp
        J = self.residual.steepest_descent_images(self.template, dW_dp)

        # Project out appearance model from VT_dW_dp
        self._J = (self.appearance_model.distance_to_subspace_vector(J.T) +
                   self.appearance_model.project_whitened_vector(J.T)).T

        # Compute Hessian and inverse
        self._H = self.residual.calculate_hessian(self._J, J2=J)

        pass

    def _align(self, max_iters=50):
        # Initial error > eps
        error = self.eps + 1

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.transform,
                                      interpolator=self._interpolator)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Request the pesudoinverse vector from the transform
            inv_delta_p = self.transform.pseudoinverse_vector(delta_p)

            # Update warp parameters
            self.transform.compose_after_from_vector_inplace(inv_delta_p)
            self.parameters.append(self.transform.as_vector())

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.transform


class ProbabilisticSymmetricCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.transform.jacobian(
            self.template.mask.true_indices)

        # Compute steepest descent images, VT_dW_dp
        self._Jt = self.residual.steepest_descent_images(self.template,
                                                         self._dW_dp)

        pass

    def _align(self, max_iters=50, alpha=1):
        # Initial error > eps
        error = self.eps + 1

        # Set relative weighting between distances
        beta = alpha - 1

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.transform,
                                      interpolator=self._interpolator)

            # Compute steepest descent images, VI_dW_dp
            Ji = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Compute symmetric steepest descent images
            J = 0.5 * (Ji + self._Jt)

            # Project out appearance model from J
            self._J = (alpha *
                       self.appearance_model.distance_to_subspace_vector(J.T) +
                       beta *
                       self.appearance_model.project_whitened_vector(J.T)).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._Jt, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Compute gradient descent parameter updates
            delta_p = 0.5 * np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            delta_transform = self.transform.from_vector(delta_p)
            delta_transform.compose_after_from_vector_inplace(delta_p)
            self.transform.compose_after_inplace(delta_transform)
            self.parameters.append(self.transform.as_vector())

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.transform


class ProbabilisticBidirectionalCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.transform.jacobian(
            self.template.mask.true_indices)

        # Compute steepest descent images, VT_dW_dp
        self._Jt = self.residual.steepest_descent_images(self.template,
                                                         self._dW_dp)

        pass

    def _align(self, max_iters=50, alpha=1):
        # Initial error > eps
        error = self.eps + 1

        # Set relative weighting between distances
        beta = alpha - 1

        # Number of shape parameters
        n_params = self.transform.n_parameters

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.transform,
                                      interpolator=self._interpolator)

            # Compute steepest descent images, VI_dW_dp
            Ji = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Compute bidirectional steepest descent images
            J = np.hstack((Ji, self._Jt))

            # Project out appearance model from J
            self._J = (alpha *
                       self.appearance_model.distance_to_subspace_vector(J.T) +
                       beta *
                       self.appearance_model.project_whitened_vector(J.T)).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._Jt, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            delta_transform = self.transform.from_vector(delta_p[:n_params])
            delta_transform.compose_after_from_vector_inplace(
                delta_p[n_params:])
            self.transform.compose_after_inplace(delta_transform)
            self.parameters.append(self.transform.as_vector())

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.transform