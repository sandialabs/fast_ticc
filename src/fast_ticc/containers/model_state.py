### Copyright 2023-2026 National Technology & Engineering Solutions of Sandia,
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

__all__ = [
    "ClusterParameters",
    "ModelState",
]

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
                 *, # All of these should be supplied by keyword
                 computed_covariance: np.ndarray,
                 empirical_covariance: np.ndarray,
                 graphical_lasso_cost: float,
                 inverse_covariance: np.ndarray,
                 log_determinant: float,
                 member_points: list[int],
                 stacked_data_mean: np.ndarray,
                 train_inverse: np.ndarray,
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
            computed_covariance=np.array([]),
            stacked_data_mean=np.array([]),
            empirical_covariance=np.array([]),
            train_inverse=np.array([]),
            inverse_covariance=np.array([]),
            log_determinant=0,
            graphical_lasso_cost=0,
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
                 *, # all arguments are keyword-only
                 arguments: arg_containers.UserArguments,
                 clusters: list[ClusterParameters],
                 label_assignment_cost: float,
                 point_labels: list[int],
                 point_log_likelihood: np.ndarray,
                 stacked_training_data: np.ndarray):

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
        empty_point_labels = list()
        return ModelState(arguments=user_args,
                          clusters=empty_cluster_info,
                          label_assignment_cost=-1,
                          point_labels=empty_point_labels,
                          point_log_likelihood=np.array([]),
                          stacked_training_data=stacked_training_data)


    def _update_cluster_membership(self):
        assert self.clusters is not None
        if (self._point_labels is None or len(self._point_labels) == 0):
            for cluster in self.clusters:
                cluster.member_points = []
        else:
            members = collections.defaultdict(list)
            for (point_id, cluster_id) in enumerate(self.point_labels):
                members[cluster_id].append(point_id)
            # We have to use range(num_clusters) here in order to pick up
            # clusters with no points in them.
            assert self.arguments is not None
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
