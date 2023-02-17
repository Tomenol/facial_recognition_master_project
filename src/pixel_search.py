"""
Description 
-----------

This script implements the Pixel comparison face recognition algorithm.
"""

# global imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from utils import *

# simulation-related parameters
threshold_norm = 961
threshold_confidence = 0.068
plotting = False

# full database
database_full = FaceDatabase(use_cache=True)

def main():
	""" Main simulation function. """

	# Face databases
	database_unknown = database_full.read_unknown()
	database_known = database_full.read_known()

	# run pixel search algorithm
	pixel_search_algorithm(database_unknown, database_known, threshold=threshold_confidence, threshold_norm=threshold_norm, plot=plotting)

	# display plots
	plt.show()

def pixel_search_algorithm(database_tests, database_ref, threshold=5, threshold_norm=961, plot=True):
	""" Performs a face recognition search based on the PCA method.

	This function researches the image closest to each image of the test database inside the reference database and 
	prints the predicted person ID compared to the person's real ID. A threshold-based identification algorithm is
	also implemented for testing purposes (Norm/Threshold).

	In this algorithm, the images are compared using their principal components projected on the eigen values of the 
	reference database.

	Parameters
	----------
		database_tests : FaceDatabase
			Test database containing the images to be identified.
		database_ref : FaceDatabase
			Reference database containing the images used as a reference for the face recognition algorithm.
		threshold : float, default=0.5
			Threshold value used for identification. If the ``confidence`` (defined as (distance - mean_distance)/mean_distance) is lower than the
			threshold, the computer is unable to identify the person id.
		threshold_norm : float, default=3.6e6
			Norm value used for identification. If the ``norm`` is lower than the threshold, the computer is unable to identify the person id.
	"""

	# indices to be plotted 
	tested_person_inds = database_tests.person_ids
	n_tested_persons = len(tested_person_inds)

	# stores the results of the face recognition algorithm
	reference_indices = np.ndarray((n_tested_persons), dtype=int) 		# True indices of the tested person
	found_indices = np.ndarray((n_tested_persons), dtype=int) 			# Indices found by the algorithm
	cost_functions = [None]*n_tested_persons 							# Cost function values

	# confidence and threshold arrays used for identification
	confidence = np.ndarray((n_tested_persons), dtype=float)
	norm = np.ndarray((n_tested_persons), dtype=float)

	# counters
	person_id = 0
	n_figs = 0

	# number of subplots per figure (used for visibility reasons)
	n_sps_per_fig = 10

	# perform the comparison between the test database and reference database
	for i in range(n_tested_persons):
		# perform the pixel to pixel comparison to search for the most probable person within the known database
		found_id, cost_function = search(database_tests.faces[i], database_ref.faces, plot=plot)

		# computes the standard deviation and mean of the cost function (recognition validity criterion)
		mean_cf = np.mean(cost_function[found_id != database_ref.person_ids])
		std_cf = np.std(cost_function[found_id != database_ref.person_ids])

		norm[i] = cost_function[found_id]
		confidence[i] = (mean_cf - cost_function[found_id])/mean_cf

		# store the data for plotting
		if database_tests.person_ids[i] in tested_person_inds:
			reference_indices[person_id] = database_ref.person_ids[found_id]
			found_indices[person_id] = database_tests.person_ids[i]
			cost_functions[person_id] = cost_function

			person_id += 1

		# Verifies if the result of the recognition is valid and compute algorithm references
		# if cost_function[found_id] <= threshold_norm:
		if confidence[i] > threshold:
			print(f"Test pixel search : {database_tests.person_ids[i]} (ref) / {database_ref.person_ids[found_id]} - FOUND")
		else:
			print(f"Test pixel search : {database_tests.person_ids[i]} (ref) / {database_ref.person_ids[found_id]} (CF : {norm[i]} | confidence : {confidence[i]}) - FAILED")

	# performance evaluation
	compute_confidence_matrix(database_full, found_indices, reference_indices, confidence, threshold)
		
	# ----------------- Plotting -----------------
	if plot is True:
		fig = plt.figure()
		ax1 = fig.add_subplot(211)
		ax2= fig.add_subplot(212)
		
		# plot norm function
		ax1.stairs(norm, fill=True, color="darkblue")
		ax1.plot([-1, len(norm)+1], [threshold_norm, threshold_norm], color="darkred")

		# plot confidence function
		ax2.stairs(confidence, fill=True, color="darkblue")
		ax2.plot([-1, len(confidence)+1], [threshold, threshold], color="darkred")

		ax1.set_ylabel(r"$C(N) = \| X(N) - X_{ref}\|_2$")
		ax2.set_ylabel(r"$F(N) = | X(N) - \bar{X}|/\bar{X}$")
		ax2.set_xlabel("Tested image")

		ax1.set_xlim([-0.5, len(norm) + 0.5])
		ax2.set_xlim([-0.5, len(norm) + 0.5])

		ax1.tick_params(labelbottom=False)
			
		# plot images
		for i in range(len(reference_indices)):
			if i//n_sps_per_fig >= n_figs: # if the number of faces per plot has been reached, create a new figure
				n_figs += 1

				# number of faces per figure needed
				n_sp = min([n_sps_per_fig, abs(len(reference_indices) - (n_figs - 1)*n_sps_per_fig)])

				# create new figure for plotting recognition results
				fig = plt.figure()

				ax = fig.subplots(2, n_sp)
				ax[0][0].set_ylabel("Reference")
				ax[1][0].set_ylabel("Result")

				# create new figure for plotting recognition cost functions
				fig = plt.figure()
				ax_criterion = fig.add_subplot(111)

			# compute reduced index
			ind = i - (n_figs - 1)*n_sps_per_fig
			found_id = np.argmin(cost_functions[i])

			# plot images
			ax[0][ind].imshow(np.reshape(database_tests.faces[i], (database_ref.n_px_v[0], database_ref.n_px_h[0])), cmap="gray") 
			ax[1][ind].imshow(np.reshape(database_ref.faces[found_id], (database_ref.n_px_v[0], database_ref.n_px_h[0])), cmap="gray")

			ax[1][ind].set_yticklabels([])
			ax[1][ind].set_xticklabels([])
			ax[0][ind].set_yticklabels([])
			ax[0][ind].set_xticklabels([])
			ax[0][ind].set_xlabel("ID " + str(found_indices[i]))
			ax[1][ind].set_xlabel("ID " + str(reference_indices[i]))

			# Plot cost function
			ax_criterion.plot(cost_functions[i], "-", label="ID" + str(reference_indices[i]))
			ax_criterion.plot(found_id, cost_functions[i][found_id], "+", color="darkred")

			ax_criterion.set_xlabel("$N$ [$-$]")
			ax_criterion.set_ylabel(r"$C(N) = \| X(N) - X_{ref}\|^2$")
			ax_criterion.legend()

if __name__ == "__main__":
	main()