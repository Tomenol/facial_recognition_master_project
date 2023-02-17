""" Module : PCA utilitary functions. """

# global imports
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# ---------------------------------------------
# 				  Functions
# ---------------------------------------------

def compute_pca(database, threshold=0.8, plot=False):
	""" Computes the Principal Components  of a given database using its own eigen vector basis.
		
	This function allows one to simply compute the Principal Components (PC) of a given database. 

	First, the algorithm computes the SVD of the faces database. To reduce the dimensionnality, only a fraction 
	``threshold`` of the information will be kept. PC are computed by projecting the database on the remaining
	eigen vectors. The PC will be stored within the database.

	Parameters
	----------
		database : FaceDatabase
			Face database which Principal Components will be calculated.
		threshold : float, default=0.8
			Percentage of information (based on the cumulative squared value of singular values) kept
			after computing the SVD of the database before computing the PC.  
		plot : bool, default=False
			If True, the results of the PC computations will be plotted.
	"""	
	print("Computing PC")

	# Substract images mean value 
	faces = database.faces - np.mean(database.faces, axis=1)[:, None]

	# if caching is enabled, look for svd results inside the root folder
	if database.use_cache is True:
		if pathlib.Path(database.root + "svd_d.npy").is_file():
			database.D = np.load(database.root + "svd_d.npy")
		if pathlib.Path(database.root + "svd_vt.npy").is_file():
			database.Vt = np.load(database.root + "svd_vt.npy")

	# if no files found or if caching disabled, compute the database svd 
	if (not (pathlib.Path(database.root + "svd_d.npy").is_file() and pathlib.Path(database.root + "svd_vt.npy").is_file())) or database.use_cache is False:
		database.U, database.D, database.Vt = np.linalg.svd(faces) # SVD computation

		# cache results
		if database.use_cache is True:
			np.save(database.root + "svd_vt.npy", database.Vt)
			np.save(database.root + "svd_d.npy", database.D)

	# compute singular values cumulative participation to the total information 
	sv_norm = np.sum(database.D**2)
	cumulative_sv_importance = np.array([np.sum(database.D[:i]**2)/sv_norm for i in range(len(database.D)+1)])
	
	# compute the index corresponding to the information cutoff threshold
	sv_cutoff_index = np.argmin(cumulative_sv_importance <= threshold)
	database.Vt = database.Vt[:sv_cutoff_index, :]

	# compute pca database
	database.pca = faces @ database.Vt.T

	# Potting
	if plot is True:
		# plotting cumulative information contained within the PCs
		fig = plt.figure()
		ax = fig.add_subplot(111)

		ax.plot(cumulative_sv_importance, "-", color="darkblue")
		ax.plot(sv_cutoff_index, cumulative_sv_importance[sv_cutoff_index], "o", color="red")
		ax.plot([-100, sv_cutoff_index], [cumulative_sv_importance[sv_cutoff_index], cumulative_sv_importance[sv_cutoff_index]], ":", color="red")
		ax.plot([sv_cutoff_index, sv_cutoff_index], [-1, cumulative_sv_importance[sv_cutoff_index]], ":", color="red")

		ax.set_xlabel("$N$ [$-$]")
		ax.set_ylabel(r"$\sum_{i=1}^{N} \sigma_i^2 / \sum_{i=1}^{N_{tot}} \sigma_i^2$ [$-$]")

		ax.set_xlim([-0.05*len(database.D), 1.05*len(database.D)])
		ax.set_ylim([-0.05, 1.05])
		
		# plotting PCs
		fig = plt.figure()
		ax = fig.add_subplot(111)

		ax.imshow(database.pca, aspect='auto')
		
		ax.set_xlabel("Principal components")
		ax.set_ylabel("Images")

		database.plot_imgs(database.Vt)

	print("PC computation done")

def project_pca(database, database_basis, plot=False):
	""" Computes the Principal Components of a given database using a given eigen vector basis.
		
	This function allows one to simply compute the Principal Components of a given database by projecting the database on 
	a given basis of eigen vectors. The PC will be stored within the database.

	Parameters
	----------
		database : FaceDatabase
			Face database which Principal Components will be calculated.
		database_basis : FaceDatabase
			Face database which Eigen vector basis will be used to compute the PC.
		plot : bool, default=False
			If True, the results of the PC computations will be plotted.
	"""
	print("Projecting PCA")

	# check if the database has a eigen vector basis
	if not hasattr(database_basis, 'Vt'):
		raise AttributeError("The database used to project the PCA results has no Vt attribute. Please compute the PCA of 'database_basis' first")

	# substract image mean values
	faces = database.faces - np.mean(database.faces, axis=1)[:, None]
	
	# Project on the eigen vector basis
	database.pca = faces @ database_basis.Vt.T
		
	print("PC projection done")
	
	# plot PCs 
	if plot is True:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.imshow(database.pca, aspect='auto')
		ax.set_xlabel("Principal components")
		ax.set_ylabel("Images")