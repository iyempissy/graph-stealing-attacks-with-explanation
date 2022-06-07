import math
import torch
import torch.nn.functional as F
from scipy.special import erf
import numpy as np
import sys


class Mechanism:
	def __init__(self, eps, input_range, **kwargs):
		self.eps = eps
		self.alpha, self.beta = input_range

	def __call__(self, x):
		raise NotImplementedError


class RandomizedResponse:
	def __init__(self, eps, d):
		self.d = d
		self.q = 1.0 / (math.exp(eps) + self.d - 1)
		self.p = self.q * math.exp(eps)

	def __call__(self, y):
		print("q", self.q)
		print("p", self.p)
		pr = y * self.p + (1 - y) * self.q
		print("pr", pr)
		out = torch.multinomial(pr, num_samples=self.d)
		print("out", out)
		return F.one_hot(out.squeeze(), num_classes=self.d)


class Laplace(Mechanism):
	def __call__(self, x):
		d = x.size(1)
		sensitivity = (self.beta - self.alpha) * d
		scale = torch.ones_like(x) * (sensitivity / self.eps)
		out = torch.distributions.Laplace(x, scale).sample()
		# out = torch.clip(out, min=self.alpha, max=self.beta)
		return out


class MultiBit(Mechanism):
	def __init__(self, *args, m='best', **kwargs):
		super().__init__(*args, **kwargs)
		self.m = m

	def __call__(self, x):
		n, d = x.size()
		if self.m == 'best':
			m = int(max(1, min(d, math.floor(self.eps / 2.18))))
		elif self.m == 'max':
			m = d
		else:
			m = self.m

		# sample features for perturbation
		BigS = torch.rand_like(x).topk(m, dim=1).indices
		s = torch.zeros_like(x, dtype=torch.bool).scatter(1, BigS, True)
		del BigS

		# perturb sampled features
		em = math.exp(self.eps / m)
		p = (x - self.alpha) / (self.beta - self.alpha)
		p = (p * (em - 1) + 1) / (em + 1)
		t = torch.bernoulli(p)
		x_star = s * (2 * t - 1)
		del p, t, s

		# unbiase the result
		x_prime = d * (self.beta - self.alpha) / (2 * m)
		x_prime = x_prime * (em + 1) * x_star / (em - 1)
		x_prime = x_prime + (self.alpha + self.beta) / 2
		return x_prime


class Gaussian(Mechanism):
	def __init__(self, *args, delta=1e-10, **kwargs):
		super().__init__(*args, **kwargs)
		self.delta = delta
		self.sigma = None
		self.sensitivity = None

	def __call__(self, x):
		len_interval = self.beta - self.alpha
		if torch.is_tensor(len_interval) and len(len_interval) > 1:
			self.sensitivity = torch.norm(len_interval, p=2)
		else:
			d = x.size(1)
			self.sensitivity = len_interval * math.sqrt(d)

		print("len_interval", len_interval)
		print("sensitivity", self.sensitivity)
		print("d", d)

		self.sigma = self.calibrate_gaussian_mechanism()
		out = torch.normal(mean=x, std=self.sigma)
		# print("Out b4 clamp", out)
		# out = torch.clamp(out, min=self.alpha, max=self.beta)
		# print("Our after clamp", out)
		print("self.sigma", self.sigma)
		return out

	def calibrate_gaussian_mechanism(self):
		return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.eps


class AnalyticGaussian(Gaussian):
	def calibrate_gaussian_mechanism(self, tol=1.e-12):
		""" Calibrate a Gaussian perturbation for differential privacy
		using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
		Arguments:
		tol : error tolerance for binary search (tol > 0)
		Output:
		sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
		"""
		delta_thr = self._case_a(0.0)
		if self.delta == delta_thr:
			alpha = 1.0
		else:
			if self.delta > delta_thr:
				predicate_stop_DT = lambda s: self._case_a(s) >= self.delta
				function_s_to_delta = lambda s: self._case_a(s)
				predicate_left_BS = lambda s: function_s_to_delta(s) > self.delta
				function_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) - math.sqrt(s / 2.0)
			else:
				predicate_stop_DT = lambda s: self._case_b(s) <= self.delta
				function_s_to_delta = lambda s: self._case_b(s)
				predicate_left_BS = lambda s: function_s_to_delta(s) < self.delta
				function_s_to_alpha = lambda s: math.sqrt(1.0 + s / 2.0) + math.sqrt(s / 2.0)
			predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - self.delta) <= tol
			s_inf, s_sup = self._doubling_trick(predicate_stop_DT, 0.0, 1.0)
			s_final = self._binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
			alpha = function_s_to_alpha(s_final)
		sigma = alpha * self.sensitivity / math.sqrt(2.0 * self.eps)
		print()
		return sigma

	@staticmethod
	def _phi(t):
		return 0.5 * (1.0 + erf(t / math.sqrt(2.0)))

	def _case_a(self, s):
		return self._phi(math.sqrt(self.eps * s)) - math.exp(self.eps) * self._phi(-math.sqrt(self.eps * (s + 2.0)))

	def _case_b(self, s):
		return self._phi(-math.sqrt(self.eps * s)) - math.exp(self.eps) * self._phi(-math.sqrt(self.eps * (s + 2.0)))

	@staticmethod
	def _doubling_trick(predicate_stop, s_inf, s_sup):
		while not predicate_stop(s_sup):
			s_inf = s_sup
			s_sup = 2.0 * s_inf
		return s_inf, s_sup

	@staticmethod
	def _binary_search(predicate_stop, predicate_left, s_inf, s_sup):
		s_mid = s_inf + (s_sup - s_inf) / 2.0
		while not predicate_stop(s_mid):
			if predicate_left(s_mid):
				s_sup = s_mid
			else:
				s_inf = s_mid
			s_mid = s_inf + (s_sup - s_inf) / 2.0
		return s_mid


class Piecewise(Mechanism):
	def __call__(self, x):
		# normalize x between -1,1
		t = (x - self.alpha) / (self.beta - self.alpha)
		t = 2 * t - 1

		# piecewise mechanism's variables
		P = (math.exp(self.eps) - math.exp(self.eps / 2)) / (2 * math.exp(self.eps / 2) + 2)
		C = (math.exp(self.eps / 2) + 1) / (math.exp(self.eps / 2) - 1)
		L = t * (C + 1) / 2 - (C - 1) / 2
		R = L + C - 1
		# print("R", R)

		# thresholds for random sampling
		threshold_left = P * (L + C) / math.exp(self.eps)
		threshold_right = threshold_left + P * (R - L)

		# masks for piecewise random sampling
		x = torch.rand_like(t)
		mask_left = x < threshold_left
		mask_middle = (threshold_left < x) & (x < threshold_right)
		mask_right = threshold_right < x

		# random sampling
		t = mask_left * (torch.rand_like(t) * (L + C) - C)
		t += mask_middle * (torch.rand_like(t) * (R - L) + L)
		t += mask_right * (torch.rand_like(t) * (C - R) + R)

		# unbias data
		x_prime = (self.beta - self.alpha) * (t + 1) / 2 + self.alpha
		return x_prime


class MultiDimPiecewise(Piecewise):
	def __call__(self, x):
		n, d = x.size()
		# print("n", n, "d", d)
		# print("math.floor(self.eps / 2.5)", math.floor(self.eps / 2.5))
		k = int(max(1, min(d, math.floor(self.eps / 2.5))))
		print("k", k)
		sample = torch.rand_like(x).topk(k, dim=1).indices
		mask = torch.zeros_like(x, dtype=torch.bool)
		mask.scatter_(1, sample, True)
		self.eps /= k
		y = super().__call__(x)
		z = mask * y * d / k
		return z


def projection(res):
	"""clips the results by projecting numbers <0.5 to 0 and >0.5 to 1"""
	res = (res > 0.5).int()
	return res


def split_explanation(x, num_elem_to_preserve, eps=0.0001, input_range=[0, 1], defense_type=1):
	"""x is the input and num_elem_to_preserve is the number of element to preserve in each splits
	eps = epsilon and input_range is the range of the transformation """
	print("||||========== Epsilon ========||||", eps)

	if defense_type == 1:

		if num_elem_to_preserve <= 1 or num_elem_to_preserve >= len(x[0]):
			raise ValueError("Element in each split should be grater than 1 and less than the feature dimension")

		print("Defense: Splitting explanations")
		# convert to numpy array
		# x = x.cpu().detach().numpy()
		# print(x)
		# res_split = np.array_split(x, num_elem_to_preserve)
		# split each row of x tensor into desired split
		x_star = []
		for i in range(0, len(x)):
			# This splits each feature vector evenly!
			res_split = torch.split(x[i], num_elem_to_preserve)  # step 1 #if you use split, it is the number of elements in each splits
			res_split = list(res_split)
			# print("res_split", res_split)
			# Want to add trailing 0 and preceeding 0 to each vector
			each_split = []
			for j in range(0, len(
					res_split)):  # step 2 and 3 #add trailing and pre 0's to "broken" or each explanations. The shape of each broken explanation should be the same as the normal ones

				rand_vec = torch.rand(1, abs(len(x[0]) - len(res_split[j])))
				zeros = torch.zeros(rand_vec.shape)
				# print("zeros==", zeros.shape)
				# print("res_split[j]", res_split[j].shape)
				if j == 0:  # extend vector after
					res_split[j] = torch.cat((res_split[j].unsqueeze(0), zeros), 1)
				elif j == len(res_split) - 1:  # extend vector before
					res_split[j] = torch.cat((zeros, res_split[j].unsqueeze(0)), 1)
				else:  # add zeros before and after
					pre_rand_vec = torch.rand(1, num_elem_to_preserve * j)
					pre_zeros = torch.zeros(pre_rand_vec.shape)

					# print("pre_zeros", pre_zeros)
					after_rand_vec = torch.rand(1, abs(len(x[0]) - ((num_elem_to_preserve * j) + len(res_split[j]))))
					after_zeros = torch.zeros(after_rand_vec.shape)

					# print("after_zeros", after_zeros)

					res_split[j] = torch.cat((pre_zeros, res_split[j].unsqueeze(0), after_zeros), 1)

				# print("j", j, "res_split[j]===", res_split[j])
				# Step 4: Do multi-dimensional Piece-wise mechanism on each split
				dp_multi_pw_transform = MultiDimPiecewise(eps, input_range)
				res_multi_pw = dp_multi_pw_transform(res_split[j])
				# print("multi dimensional piecewise",
				#       res_multi_pw)  # multi dimensional piecewise tensor([[0.0000, 5.0558, 0.0000]])

				projected_res_multi_pw = projection(res_multi_pw)
				# print("Projection==", projected_res_multi_pw)

				each_split.append(projected_res_multi_pw)

			# add them together
			added_split = torch.stack(each_split, dim=0).sum(dim=0)
			x_star.append(added_split)

		# print("x_star", x_star)

		all_split = torch.cat(x_star)
		all_split = projection(all_split)  # To make it 0 and 1 after adding!
		print("all_split", all_split)
		return all_split.float()

	elif defense_type == 2:  # Do multi dimensional piecewise mechanism
		print("Only doing MultiDimPiecewise mechanism!")
		dp_multi_pw_transform = MultiDimPiecewise(eps, input_range)
		res_multi_pw = dp_multi_pw_transform(x)

		projected_res_multi_pw = projection(res_multi_pw)

		print("res_multi_pw", projected_res_multi_pw)
		return projected_res_multi_pw.float()
	elif defense_type == 3:
		print("============= Doing Gaussian ==============")
		gau = AnalyticGaussian(eps, input_range)
		res_multi_pw = gau(x)
		print("res_multi_pw", res_multi_pw)
		projected_res_multi_pw = projection(res_multi_pw)
		return projected_res_multi_pw.float()
	elif defense_type == 4:
		print("=============== Multibit Mechanism===========")
		multi_bit = MultiBit(eps, input_range)
		res_multi_pw = multi_bit(x)
		print("res_multi_pw", res_multi_pw)
		projected_res_multi_pw = projection(res_multi_pw)
		return projected_res_multi_pw.float()
	elif defense_type == 5:
		print("=============== Randomized response Mechanism===========")
		rand_resp_res = randomized_response(x, eps)
		return rand_resp_res.float()


def randomized_response(x, eps):
	dim = len(x[0])
	# q = 1.0 / (math.exp(eps) + dim - 1)
	# p = q * math.exp(eps)

	p = math.exp(eps) / (math.exp(eps) + 1)
	q = 1 - p

	print("p", p)
	print("q", q)
	print("dim", dim)
	all_x = []
	for d in x:
		each_vec = []
		for i in d:
			sample = np.random.random()
			sample2 = np.random.random()

			# print("sample", sample)
			# print("sample2", sample2)

			if sample < p:
				each_vec.append(i)  # keep original
				# return i
			else:
				if sample2 < q:  # 2nd flip, always return 1 with prob q and 0 otherwise!
					each_vec.append(torch.tensor(1))
					# return 1
				else:
					each_vec.append(torch.tensor(0))
					# return 0

		# convert from list of tensor to a single tensor
		each_vec = torch.FloatTensor(each_vec)
		# print("each_vec2", each_vec)
		all_x.append(each_vec)

	all_x = torch.stack(all_x, 0)
	print(all_x.shape)
	# print("all_x", all_x)
	return all_x


def explanation_intersection(original, perturbed):
	# count the numbers of 1's that both the original and perturb has.
	# Divide by total 1 in the original
	# add each together and divide by the len(original)
	# Multiply by 100 to get percentage
	original_count_all_tensor = torch.unique(original, sorted=True, return_counts=True)
	original_count_1_all_tensor = original_count_all_tensor[1][1].item()
	# print("original_count of 1", original_count_1_all_tensor)

	each_count_1 = []

	for i in range(len(original)):
		intersection_count = 0
		# original_count = torch.unique(original[i], sorted=True, return_counts=True)
		# original_count_1 = original_count[1][1].item()
		# # print("original_count_1", original_count_1)
		original_count_1 = 0

		# check if each explanation feature is the same
		for j in range(len(original[i])):
			if original[i][j] == 1:
				original_count_1 += 1

			if (original[i][j] == perturbed[i][j]) and original[i][j] == 1:
				# print("original[i][j]", original[i][j], "perturbed[i][j]", perturbed[i][j])
				# print(j)
				intersection_count += 1

		# print("Count",i, intersection_count)
		average_count_1 = intersection_count / original_count_1
		each_count_1.append(average_count_1)

	final_count = (sum(each_count_1) / len(each_count_1))
	percentage_count = final_count * 100
	print("percentage_count", percentage_count)
	return percentage_count


def vanilla_ldp_epsilon(p):
	# print("(1 - p)", 1 - p)
	return np.log(p / (1 - p))


def unary_epsilon(p, q=0):
	q = 1 - p
	return np.log((p * (1 - q)) / ((1 - p) * q))
