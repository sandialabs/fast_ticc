### Copyright 2023 National Technology & Engineering Solutions of Sandia,
### LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
### U.S. Government retains certain rights in this software.
###
### Redistribution and use in source and binary forms, with or without
### modification, are permitted provided that the following conditions are
### met:
###
### 1. Redistributions of source code must retain the above copyright
###    notice, this list of conditions and the following disclaimer.
###
### 2. Redistributions in binary form must reproduce the above copyright
###    notice, this list of conditions and the following disclaimer in
###    the documentation and/or other materials provided with the
###    distribution.
###
### 3. Neither the name of the copyright holder nor the names of its
###    contributors may be used to endorse or promote products derived
###    from this software without specific prior written permission.
###
### THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
### “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
### LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
### A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
### HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
### SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
### LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
### DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
### THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
### (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
### OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Containers for TICC parameters, model state, and result.

Contents:
    ClusterParameters: Covariance matrix and derived values for each cluster
    ModelState: All the data for a TICC model under construction
"""

import collections
from typing import List, Optional

import numpy as np

from fast_ticc.containers import arguments as arg_containers

class ClusterParameters:
    """Container for TICC parameters for a single cluster

    Note: NW is 'number of time series * window size', a constant that
    appears frequently in the TICC paper.

    Properties:
        computed_covariance (NumPy array):
        empirical_covariance (NumPy array): NW x NW covariance matrix for
            member data points
        graphical_lasso_cost (float): Final cost value for optimized
            MRF / graphical lasso matrix
        inverse_covariance (NumPy array): ???
        log_determinant (float): Logarithm of the determinant of the
            covariance matrix (???)
        member_points (list of int): Indices of points belonging to this
            cluster
        stacked_data_mean (NumPy array): Mean of all (stacked) data points
            belonging to this cluster
        train_inverse (NumPy array): Optimized block-Toeplitz inverse
            covariane matrix.  This is one of TICC's main results.
        """

    def __init__(self,
                 computed_covariance: Optional[np.ndarray] = None,
                 empirical_covariance: Optional[np.ndarray] = None,
                 graphical_lasso_cost: Optional[float] = None,
                 inverse_covariance: Optional[np.ndarray] = None,
                 log_determinant: Optional[np.ndarray] = None,
                 member_points: Optional[List[int]] = None,
                 stacked_data_mean: Optional[np.ndarray] = None,
                 train_inverse: Optional[np.ndarray] = None,
                 ):

        self.computed_covariance = computed_covariance
        self.empirical_covariance = empirical_covariance
        self.graphical_lasso_cost = graphical_lasso_cost
        self.inverse_covariance = inverse_covariance
        self.log_determinant = log_determinant
        self.stacked_data_mean = stacked_data_mean
        self.train_inverse = train_inverse
        if member_points is None:
            member_points = []
        self._member_points = sorted(member_points)

    @property
    def member_points(self) -> List:
        """Get the indices of the member points in this cluster.

        No arguments.

        Returns:
            Member point indices as integers.  Do not assume that these will
            be sorted.
        """
        return self._member_points

    @member_points.setter
    def member_points(self, new_members):
        if new_members is None:
            self._member_points = []
        elif len(new_members) == 0:
            self._member_points = []
        elif new_members != self._member_points:
            self._member_points = sorted(new_members)

    @property
    def size(self) -> int:
        """Number of points assigned to this cluster

        Returns:
            Length of member_points array for this cluster.
        """
        if self.member_points is not None:
            return len(self.member_points)
        return 0

    @staticmethod
    def empty_cluster() -> "ClusterParameters":
        """Allocate a new set of cluster parameters

        The cluster parameters returned from this function have
        no numerical state or member points.

        Returns:
            New set of uninitialized cluster parameters
        """

        return ClusterParameters(
            member_points=[],
            computed_covariance=None,
            stacked_data_mean=None,
            empirical_covariance=None,
            train_inverse=None,
            inverse_covariance=None,
            log_determinant=None
        )

    def shallow_copy(self) -> "ClusterParameters":
        """Shallow copy of cluster parameters

        As a shallow copy, the non-atomic data items in the result
        (the member point lists and forward/inverse covariance
        matrices) are just pointers to the originals.  If you modify
        them, you will also modify the originals.

        Returns:
            Shallow copy of cluster parameters
        """

        return ClusterParameters(
            computed_covariance=self.computed_covariance,
            empirical_covariance=self.empirical_covariance,
            graphical_lasso_cost=self.graphical_lasso_cost,
            inverse_covariance=self.inverse_covariance,
            log_determinant=self.log_determinant,
            member_points=self.member_points,
            stacked_data_mean=self.stacked_data_mean,
            train_inverse=self.train_inverse
        )

    def deep_copy(self) -> "ClusterParameters":
        """Deep copy of cluster parameters

        As a deep copy, the return value from this function can be
        modified without affecting the original.

        No arguments.

        Returns:
            New copy of cluster parameters
        """
        return ClusterParameters(
            computed_covariance=np.copy(self.computed_covariance),
            empirical_covariance=np.copy(self.empirical_covariance),
            graphical_lasso_cost=self.graphical_lasso_cost,
            inverse_covariance=np.copy(self.inverse_covariance),
            log_determinant=self.log_determinant,
            member_points=list(self.member_points),
            stacked_data_mean=np.copy(self.stacked_data_mean),
            train_inverse=np.copy(self.train_inverse)
        )


class ModelState:
    """All state information for a TICC model in progress.

    In this class, D means 'number of data points', N means 'number of
    time series' (values at each data point), and W means 'TICC window
    size'.  K is the number of clusters.

    Properties:
        arguments (UserParameters): User-specified parameters to the TICC
            algorithm such as beta and lambda values, number of clusters,
            maximum number of iterations, and so on.
        clusters (list of ClusterParameters): Matrices and membership list
            for each cluster being built
        label_assignment_cost (float): Total cost of the latest set of
            cluster labels for points.  This is the value of equation 3
            in the TICC paper.
        point_labels (list of integers): Cluster labels for each point.
            This information should always be the same as what's stored
            in the collection of the member_points arrays in the clusters.
        point_log_likelihood: D x K NumPy array.  Each value is the
            negative log likelihood of one data point with respect to
            one particular cluster.
        stacked_training_data (NumPy array): D x NW array of values.  This
            is the result of stacking W copies of the input data vertically
            with each subsequent copy shifted forward one time step.  This
            array contains the points named X<sub>t</sub> from the TICC paper.
        """

    def __init__(self,
                 arguments: Optional[arg_containers.UserArguments] = None,
                 clusters: Optional[List[ClusterParameters]] = None,
                 label_assignment_cost: Optional[float] = None,
                 point_labels: Optional[List[int]] = None,
                 point_log_likelihood: Optional[np.ndarray] = None,
                 stacked_training_data: Optional[np.ndarray] = None):

        self.arguments = arguments
        self.clusters = clusters
        self.label_assignment_cost = label_assignment_cost
        self._point_labels = point_labels
        self.point_log_likelihood = point_log_likelihood
        self.stacked_training_data = stacked_training_data

    @staticmethod
    def empty_model(user_args: arg_containers.UserArguments,
                    stacked_training_data: np.ndarray) -> "ModelState":
        """Create an empty TICC model from arguments and training data.

        The return value from this function will contain the supplied
        user arguments and training data.  None of the numerical state
        about the clusters or their labels will be filled in.

        Arguments:
            user_args (fast_ticc.containers.arguments.UserArguments):
                User arguments to clusteirng process
            stacked_training_data (numpy.ndarray): Data for training
                model

        Returns:
            Freshly allocated
            fast_ticc.containers.model_state.ModelState instance
        """
        empty_cluster_info = [
            ClusterParameters.empty_cluster()
            for i in range(user_args.num_clusters)
            ]
        return ModelState(arguments=user_args,
                          clusters=empty_cluster_info,
                          stacked_training_data=stacked_training_data)


    def _update_cluster_membership(self):
        if (self._point_labels is None or len(self._point_labels) == 0):
            for cluster in self.clusters:
                cluster.member_points = []
        else:
            members = collections.defaultdict(list)
            for (point_id, cluster_id) in enumerate(self.point_labels):
                members[cluster_id].append(point_id)
            # We have to use range(num_clusters) here in order to pick up
            # clusters with no points in them.
            for cluster_id in range(self.arguments.num_clusters):
                this_cluster_members = members[cluster_id]
                self.clusters[cluster_id].member_points = this_cluster_members


    @property
    def point_labels(self) -> List[int]:
        """Cluster labels for each point in the model

        This is the latest set of labels computed by TICC.  Label values
        range from -1 (point not labeled) to num_clusters - 1.
        """
        return self._point_labels

    @point_labels.setter
    def point_labels(self, new_labels: List[int]):
        if new_labels != self._point_labels:
            self._point_labels = new_labels
            self._update_cluster_membership()

    def shallow_copy(self) -> "ModelState":
        """Create a shallow copy of this model's state.

        Since this is a shallow copy, the arguments, cluster parameters,
        log likelihood data, and training data are just pointers to the
        originals.  Modifying them will modify the original as well.

        No arguments.

        Returns:
            Shallow copy of model state.
        """

        return ModelState(
            arguments=self.arguments,
            clusters=list(self.clusters),
            label_assignment_cost=self.label_assignment_cost,
            point_labels=self._point_labels,
            point_log_likelihood=self.point_log_likelihood,
            stacked_training_data=self.stacked_training_data
        )

    def deep_copy(self) -> "ModelState":
        """Create a deep copy of this model's state.

        Since this is a deep copy, all data will be copied and none will be
        shared with the original.  You can modify the return value without
        affecting the model that you copied.

        No arguments.

        Returns:
            New copy of model state.
        """

        new_clusters = [cluster.deep_copy() for cluster in self.clusters]
        return ModelState(
            arguments=self.arguments.deep_copy(),
            clusters=new_clusters,
            label_assignment_cost=self.label_assignment_cost,
            point_labels=list(self._point_labels),
            point_log_likelihood=np.copy(self.point_log_likelihood),
            stacked_training_data=np.copy(self.stacked_training_data)
        )
