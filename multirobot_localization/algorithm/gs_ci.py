"""GS-CI local filter with class-conditional process/sensor scaling."""

from math import cos, sin

import numpy as np
from numpy import matrix
from scipy import linalg

import sim_env
from agent_classes import DEFAULT_AGENT_CLASS_PROFILES, AgentClass, normalize_agent_class


class GS_CI:

	def __init__(
		self,
		_index,
		_initial_s,
		_theta=0.0,
		agent_class=AgentClass.CLASS_A_UGV,
		epsilon=0.1,
		class_quantiles=None,
		ci_coeff=0.8,
		class_profile=None,
	):
		self.index = _index
		self.s = _initial_s.copy()
		self.sigma = sim_env.initial_cov.copy()
		self.th_sigma = sim_env.initial_cov.copy()
		self.theta = _theta
		self.agent_class = normalize_agent_class(agent_class)
		self.epsilon = float(epsilon)
		self.class_quantiles = self._normalize_quantile_map(class_quantiles)
		self.ci_coeff = float(ci_coeff)
		self.class_profile = class_profile or DEFAULT_AGENT_CLASS_PROFILES[self.agent_class]

		# Class profile values scale motion and measurement uncertainty.
		self.var_u_v = sim_env.var_u_v * self.class_profile.process_var_scale
		self.var_v = sim_env.var_v * self.class_profile.unobserved_process_var_scale
		self.var_dis = sim_env.var_dis * self.class_profile.range_var_scale
		self.var_phi = sim_env.var_phi * self.class_profile.bearing_var_scale

	def _normalize_quantile_map(self, class_quantiles):
		if class_quantiles is None:
			return {}
		normalized = {}
		for key, value in class_quantiles.items():
			normalized[normalize_agent_class(key)] = float(value)
		return normalized

	def set_class_quantiles(self, class_quantiles):
		self.class_quantiles = self._normalize_quantile_map(class_quantiles)

	def get_class_quantile(self, agent_class=None, class_quantiles=None):
		quantiles = self.class_quantiles.copy()
		quantiles.update(self._normalize_quantile_map(class_quantiles))
		agent_class = normalize_agent_class(agent_class or self.agent_class)
		return float(quantiles.get(agent_class, 1.0))

	def _symmetrize(self, sigma):
		sigma_array = np.asarray(sigma, dtype=float)
		return matrix(0.5 * (sigma_array + sigma_array.T))

	def _stable_information(self, sigma):
		sigma_array = np.asarray(self._symmetrize(sigma), dtype=float)
		return matrix(linalg.pinvh(sigma_array))

	def _calibrated_covariance(self, sigma, agent_class=None, class_quantiles=None):
		quantile = self.get_class_quantile(agent_class=agent_class, class_quantiles=class_quantiles)
		return self._symmetrize(float(quantile) * np.asarray(sigma, dtype=float))

	def _sender_orientation_mask(self):
		# TODO: Replace the identity mask if the state is extended to include
		# sender-specific orientation components as in the paper notation.
		return matrix(np.eye(self.s.shape[0], dtype=float))

	def _receiver_orientation_mask(self):
		# TODO: Replace the identity insertion matrix if the receiver state is
		# augmented with an explicit orientation state.
		return matrix(np.eye(self.s.shape[0], dtype=float))

	def motion_propagation_update(self, odometry_input, dt):

		[v, omega] = odometry_input
		ii = 2 * self.index

		# estimation update
		self.s[ii, 0] = self.s[ii, 0] + cos(self.theta) * v * dt
		self.s[ii + 1, 0] = self.s[ii + 1, 0] + sin(self.theta) * v * dt

		# Covariance propagation updates self and non-self blocks differently.
		for j in range(sim_env.N):
			jj = 2 * j

			if j == self.index:
				rot_mtx_theta = sim_env.rot_mtx(self.theta)
				self.sigma[jj:jj + 2, jj:jj + 2] += (dt ** 2) * rot_mtx_theta * matrix([[self.var_u_v, 0], [0, 0]]) * rot_mtx_theta.T
				self.th_sigma[jj:jj + 2, jj:jj + 2] += 2 * (dt ** 2) * matrix([[self.var_u_v, 0], [0, 0]])

			else:
				self.sigma[jj:jj + 2, jj:jj + 2] += (dt ** 2) * self.var_v * sim_env.i_mtx_2.copy()
				self.th_sigma[jj:jj + 2, jj:jj + 2] += (dt ** 2) * self.var_v * sim_env.i_mtx_2.copy()

	def ablt_obsv_update(self, obs_value, landmark):
		ii = 2 * self.index

		H_i = matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
		H_i[0, ii] = -1
		H_i[1, ii + 1] = -1

		H = sim_env.rot_mtx(self.theta).getT() * H_i

		dis = obs_value[0]
		phi = obs_value[1]

		hat_z = sim_env.rot_mtx(self.theta).getT() * (landmark.position + H_i * self.s)
		z = matrix([dis * cos(phi), dis * sin(phi)]).getT()

		# Absolute landmark update in Cartesianized measurement space.
		sigma_z = sim_env.rot_mtx(phi) * matrix([[self.var_dis, 0], [0, (dis ** 2) * self.var_phi]]) * sim_env.rot_mtx(phi).getT()
		sigma_invention = H * self.sigma * H.getT() + sigma_z
		kalman_gain = self.sigma * H.getT() * sigma_invention.getI()

		self.s = self.s + kalman_gain * (z - hat_z)
		self.sigma = self._symmetrize(self.sigma - kalman_gain * H * self.sigma)

		sigma_th_z = max(self.var_dis, (sim_env.d_max ** 2) * self.var_phi) * sim_env.i_mtx_2.copy()
		self.th_sigma = self._symmetrize((self.th_sigma.getI() + H_i.getT() * sigma_th_z.getI() * H_i).getI())

	def rela_obsv_update(self, obs_idx, obs_value):
		ii = 2 * self.index
		jj = 2 * obs_idx

		H_ij = matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
		H_ij[0, ii] = -1
		H_ij[1, ii + 1] = -1
		H_ij[0, jj] = 1
		H_ij[1, jj + 1] = 1

		H = sim_env.rot_mtx(self.theta).getT() * H_ij

		dis = obs_value[0]
		phi = obs_value[1]

		hat_z = H * self.s
		z = matrix([dis * cos(phi), dis * sin(phi)]).getT()

		# Relative inter-robot update.
		sigma_z = sim_env.rot_mtx(phi) * matrix([[self.var_dis, 0], [0, (dis ** 2) * self.var_phi]]) * sim_env.rot_mtx(phi).getT()
		sigma_invention = H * self.sigma * H.getT() + sigma_z
		kalman_gain = self.sigma * H.getT() * sigma_invention.getI()

		# update
		sigma_th_z = max(self.var_dis, (sim_env.d_max ** 2) * self.var_phi) * sim_env.i_mtx_2.copy()
		self.th_sigma = self._symmetrize((self.th_sigma.getI() + H_ij.getT() * sigma_th_z.getI() * H_ij).getI())
		self.s = self.s + kalman_gain * (z - hat_z)
		self.sigma = self._symmetrize(self.sigma - kalman_gain * H * self.sigma)

	def comm_update(
		self,
		comm_robot_s,
		comm_robot_sigma,
		comm_robot_th_sigma,
		comm_robot_class=None,
		class_quantiles=None,
		ci_coeff=None,
	):
		"""Fuse a communicated estimate using class-conditional calibrated CI.

		The current GS-CI simulator stores only planar positions in ``self.s``.
		Therefore the orientation masking matrices from the paper reduce to the
		identity by default. The helper hooks remain in place for a future
		orientation-augmented state implementation.
		"""

		ci_coeff = self.ci_coeff if ci_coeff is None else float(ci_coeff)
		remote_state = matrix(comm_robot_s, dtype=float)
		remote_class = normalize_agent_class(comm_robot_class or self.agent_class)

		quantiles = self.class_quantiles.copy()
		quantiles.update(self._normalize_quantile_map(class_quantiles))

		local_sigma_tilde = self._calibrated_covariance(self.sigma, agent_class=self.agent_class, class_quantiles=quantiles)
		remote_sigma_tilde = self._calibrated_covariance(comm_robot_sigma, agent_class=remote_class, class_quantiles=quantiles)

		T_j_minus = self._sender_orientation_mask()
		T_i_plus = self._receiver_orientation_mask()

		remote_reduced_sigma = self._symmetrize(T_j_minus * remote_sigma_tilde * T_j_minus.T)
		remote_information_reduced = self._stable_information(remote_reduced_sigma)
		incoming_information = self._symmetrize(T_i_plus * remote_information_reduced * T_i_plus.T)
		incoming_information_vector = T_i_plus * remote_information_reduced * T_j_minus * remote_state

		local_information = self._stable_information(local_sigma_tilde)
		local_information_vector = local_information * self.s

		fused_information = self._symmetrize(ci_coeff * local_information + (1 - ci_coeff) * incoming_information)
		fused_covariance = self._symmetrize(self._stable_information(fused_information))
		fused_state = fused_covariance * (
			ci_coeff * local_information_vector + (1 - ci_coeff) * incoming_information_vector
		)

		self.s = fused_state
		self.sigma = fused_covariance

		q_sup = 1.0
		if quantiles:
			q_sup = max(float(value) for value in quantiles.values())
		local_th_sigma_tilde = self._symmetrize(q_sup * np.asarray(self.th_sigma, dtype=float))
		remote_th_sigma_tilde = self._symmetrize(q_sup * np.asarray(comm_robot_th_sigma, dtype=float))
		fused_th_information = self._symmetrize(
			ci_coeff * self._stable_information(local_th_sigma_tilde)
			+ (1 - ci_coeff) * self._stable_information(remote_th_sigma_tilde)
		)
		self.th_sigma = self._symmetrize(self._stable_information(fused_th_information))
