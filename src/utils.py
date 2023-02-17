""" Convenience classes and functions. """

# global imports
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

import copy
import pickle

import pathlib
import tqdm

# local imports
import make_pca



# ----------------------------------
# 			   Classes
# ----------------------------------

class FaceDatabase(object):
	""" Encapsulates a database of Faces. 

	This class is used to store and manage a set of reference faces extracted from a catalog of images. 
	"""
	def __init__(self, use_cache=True, load_default_catalog=True, root="data", database_dir="./../data"):
		""" Default class constructor 
	
		Intializes the database of reference faces.

		Parameters 
		----------
			use_cache : bool, default=True
				If True, the content and main results of the calculations done within the class will be stored 
				within the ``root`` folder.
			load_default_catalog : bool, default=True
				If True, the default catalog (located at ``database_dir``) will be loaded on initialization.
			root : str, default="data"
				Working root directory.
			database_dir : str, default="./../data"
				Database location.
		"""
		
		# If root could not be located, create a new root directory
		if not pathlib.Path(root).is_dir():
			os.mkdir(root)

		# Class initialization
		self.root = root + "/" 
		self.use_cache = use_cache

		self.pca = None
		self.face_database_dir = self.root + "faces_database.npy"	
		self.face_metadata_dir = self.root + "faces_metadata.pickle"	
		
		# if enabled, load the catalog
		if load_default_catalog is True:
			self.database_dir = database_dir # default databalse location
			
			# if the files are already cached on the drive, load them
			if pathlib.Path(self.face_database_dir).is_file() and pathlib.Path(self.face_metadata_dir).is_file() and self.use_cache is True:
				print("Loading face database from cached np array : ", self.face_database_dir)
				
				# load faces and metadata
				self.faces = np.load(self.face_database_dir)
				metadata = self.__load_pickle(self.face_metadata_dir)

				self.homogeneous_arrays = metadata["homogeneous_arrays"]
				self.faces = metadata["faces"]
				self.person_ids = metadata["person_ids"]
				self.img_ids = metadata["img_ids"]
				self.n_pictures_per_person = metadata["n_pictures_per_person"]
				self.n_pixels = metadata["n_pixels"]
				self.n_px_h = metadata["n_px_h"]
				self.n_px_v = metadata["n_px_v"]
				self.n_persons = metadata["n_persons"]
			else: 
				print("Loading face database from data folder.")

				# if the files do not already exist, reads the default catalog and saves the content of the class on the drive
				self.read_catalog(self.database_dir)

			print("Done")

	def compute_pca(self, threshold=0.8, plot=False):
		""" Computes the Principal Components of the database using its own eigen vector basis.
		
		This function allows one to simply compute the Principal Components (PC) of the database. 

		First, the algorithm computes the SVD of the faces database. To reduce the dimensionnality, only a fraction 
		``threshold`` of the information will be kept. The PC is computed by projecting the database on the remaining
		eigen vectors. The PC will be stored within the database.

		Parameters
		----------
			threshold : float, default=0.8
				Percentage of information (based on the cumulative squared value of singular values) kept
				after computing the SVD of the database before computing the PC.  
			plot : bool, default=False
				If True, the results of the PC computations will be plotted.
		"""
		make_pca.compute_pca(self, threshold=threshold, plot=plot)

	def project_pca(self, database_basis, plot=False):
		""" Computes the Principal Components of the database using a given eigen vector basis.
		
		This function allows one to simply compute the Principal Components of the database by projecting the database on 
		a given basis of eigen vectors. The PC will be stored within the database.

		Parameters
		----------
			database_basis : numpy.ndarray (N, M)
				Eigen vector basis used to compute the PC.
			plot : bool, default=False
				If True, the results of the PC computations will be plotted.
		"""
		make_pca.project_pca(self, database_basis, plot=plot)

	def read_catalog(self, database_dir, extension="pgm"):
		""" Reads a catalog of faces located in a given folder.
		
		Imports a set of reference faces from a specified data folder and store it within the database class.

		Parameters
		----------
			database_dir : str
				Path from which the data will be imported. 
			extension : str, default='pgm'
				Image file extensions.
		"""
		# convert the string to a path
		path = pathlib.Path(database_dir)

		# get and sort all data file paths
		dirs = glob(str(path / "s*"))
		files = []
		person_ids_tmp = []

		for dir_ in dirs:
			tmp_files = glob(dir_ + "/*." + extension) 
			img_ids_tmp = [int(file.split('/')[-1].split('.')[0]) for file in tmp_files]

			# sort filenames according to image index
			files_sorted = [x for x, _ in sorted(zip(tmp_files, img_ids_tmp), key=lambda pair: pair[1])]
			img_ids_tmp = sorted(img_ids_tmp)

			# compute person id to sort files in 
			person_ids_tmp += [int(str(dir_).split('s')[-1]) - 1]

			# store filenames
			files += [files_sorted]

		# sort all the file paths according to their ids.
		files = [x for x, _ in sorted(zip(files, person_ids_tmp), key=lambda pair: pair[1])]
		person_ids_tmp = sorted(person_ids_tmp)

		# progress bar
		p = tqdm.tqdm(total=len(files))

		# class database variables
		self.faces = []
		self.person_ids = []
		self.n_px_h = []
		self.n_px_v = []
		self.n_pictures_per_person = []
		self.img_ids = []

		# import images from each sub folder
		for i, files_ in enumerate(files):
			faces_subdir = [self.read_image(x) for x in files_]

			self.n_px_v += [x.shape[0] for x in faces_subdir]
			self.n_px_h += [x.shape[1] for x in faces_subdir]
			self.faces  += [x.flatten() for x in faces_subdir]
			self.n_pictures_per_person  += [len(faces_subdir)]

			self.person_ids += list(np.repeat(person_ids_tmp[i], len(files_)))
			self.img_ids += list(np.arange(0, len(files_)))

			p.update(1)

		del p

		# update the class state
		self.n_persons = len(files) 
		self.n_pixels = np.prod([self.n_px_v, self.n_px_h], axis=0)

		# if the arrays are homogeneous (i.e. same number of pixels per images), all the data is converted to matrices and 1D vectors (numpy arrays)
		if len(np.unique(self.n_pixels)) == 1:
			self.homogeneous_arrays = True
			
			self.faces = np.array(self.faces)
			self.person_ids = np.array(self.person_ids)
			self.img_ids = np.array(self.img_ids)
			
			self.n_pictures_per_person = np.array(self.n_pictures_per_person)
			self.n_pixels = np.array(self.n_pixels)
			self.n_px_h = np.array(self.n_px_h)
			self.n_px_v = np.array(self.n_px_v)
		else:
			self.homogeneous_arrays = False
			print("Warning : arrays are not homogeneous. This code has not yet been tested for inhomogeneous databases")

		# if cacheing is enabled, the data is stored as numpy arrays and pickles
		if self.use_cache is True:
			print("Saving face data at ", self.face_database_dir)

			np.save(self.face_database_dir, self.faces)

			metadata = dict()
			metadata["homogeneous_arrays"] = self.homogeneous_arrays
			metadata["faces"] = self.faces
			metadata["person_ids"] = self.person_ids
			metadata["img_ids"] = self.img_ids
			metadata["n_pictures_per_person"] = self.n_pictures_per_person
			metadata["n_pixels"] = self.n_pixels
			metadata["n_px_h"] = self.n_px_h
			metadata["n_px_v"] = self.n_px_v
			metadata["n_persons"] = self.n_persons

			self.__save_pickle(self.face_metadata_dir, metadata)

	def read_image(self, filename):
		""" Imports an image file located at a given location.
	
		Reads an input image and converts it to a numpy array.

		Parameters
		----------
			filename : str
				Image file path.

		Returns
		-------
			numpy.ndarray (N, M): 
				image data. 
		"""
		# open the image file
		img_file = open(filename, 'rb')
		if img_file is None:
			raise ValueError(f"Image file {img_file} could not be found.")

		# read the image
		image = plt.imread(img_file)
		
		return image

	def extract_images(self, exclude_person=[], exclude_id=[], mask=None):
		""" Extracts a set of images from the database.
	
		This function generates a new database based the current database by removing a set of 
		images.

		Parameters
		----------
			exclude_person : list, default=[]
				Indices of the persons to remove from the database.
			exclude_id : list, default=[]
				Indices of the images to remove from the database.
			mask : numpy.ndarray (N, M), default=None
				Binary mask specifying which images to remove from the database.

		Returns
		-------
			FaceDatabase : 
				New database stripped from the specified images.
		"""
		# if no mask is provided create a mask from the specified ids to remove
		if mask is not None:
			# check of id arrays are empty
			if np.size(exclude_person) > 0 or np.size(exclude_id) > 0:
				raise ValueError("Either specify ids or exclusion masks, but not both.")
		else:
			mask = np.full(len(self.faces), True)
			
			# find all images which are to be removed
			if np.size(exclude_person) > 0:
				mask_persons = np.logical_and.reduce([self.person_ids != id_ for id_ in exclude_person])
				mask = np.logical_and(mask, mask_persons)
			if np.size(exclude_id) > 0:
				mask_id = np.logical_and.reduce([self.img_ids != id_ for id_ in exclude_id])
				mask = np.logical_and(mask, mask_id)

		# remove images from database and update database values.
		if self.homogeneous_arrays is True:
			new_database = copy.copy(self)

			new_database.faces = self.faces[mask]
			new_database.person_ids = self.person_ids[mask]
			new_database.img_ids = self.img_ids[mask]
			new_database.n_pictures_per_person = np.array([len(new_database.img_ids[new_database.person_ids == id_]) for id_ in np.unique(new_database.person_ids)])
			new_database.n_pixels = self.n_pixels[mask]
			new_database.n_px_h = self.n_px_h[mask]
			new_database.n_px_v = self.n_px_v[mask]

			# only if the Principal Components have been computed 
			if self.pca is not None:
				new_database.pca = self.pca[mask]
		else:
			print("Warning : Image selection for non-homogeneous arrays is not tested yet.")
			
			new_database.faces = [self.faces[mask[i]] for i in range(len(self.faces))]
			new_database.person_ids = [self.person_ids[mask] for i in range(len(self.faces))]
			new_database.img_ids = [self.img_ids[mask] for i in range(len(self.faces))]
			new_database.n_pictures_per_person = np.array([len(new_database.img_ids[new_database.person_ids == id_]) for id_ in np.unique(new_database.person_ids)])
			new_database.n_pixels = [self.n_pixels[mask] for i in range(len(self.faces))]
			new_database.n_px_h = [self.n_px_h[mask] for i in range(len(self.faces))]
			new_database.n_px_v = [self.n_px_v[mask] for i in range(len(self.faces))]

			# only if the Principal Components have been computed 
			if self.pca is not None:
				new_database.pca = [self.pca[mask] for i in range(len(self.faces))]

		new_database.n_persons = len(new_database.n_pictures_per_person)

		return new_database

	def read_known(self):
		""" Generates the database of known persons from the full database.
	
		This function is specific to the default full database and allows users to generate
		the database of known persons.

		Returns
		-------
			FaceDatabase :
				Database of known persons.
		"""
		return self.extract_images(exclude_person=[39], exclude_id=[9])

	def read_unknown(self):
		""" Generates the database of unknown persons from the full database.
	
		This function is specific to the default full database and allows users to generate
		the database of unknown persons.

		Returns
		-------
			FaceDatabase :
				Database of unknown persons.
		"""
		mask = np.logical_or(self.person_ids == 39, self.img_ids == 9)
		db_tmp  = self.extract_images(mask=mask)
		return db_tmp

	def __load_pickle(self, filename):
		""" Loads and returns the data contained inside a pickle file.

		Parameters
		----------
			filename : str
				Pickle file path.

		Returns
		-------
			object :
				Data contained within the pickle file.
		"""
		file = open(filename, 'rb')
		data = pickle.load(file)

		return data

	def __save_pickle(self, filename, data):
		""" Saves the data contained to a specified a pickle file.

		Parameters
		----------
			filename : str
				Pickle file path.
			data : object
				Object / data to store within the pickle file.
		"""
		file = open(filename, 'wb')
		pickle.dump(data, file)

	def plot(self):
		""" Plots all the images contained within the database. """
		if self.homogeneous_arrays is True:
			self.plot_imgs(self.faces)
		else:
			raise NotImplementedError("Image plotting for non-homogeneous arrays is not implemented yet.")

	def plot_imgs(self, images, n_rows=None, n_cols=None):
		""" Plots the given set of images.

		Parameters
		----------
			images : numpy.ndarray
				Array which lines contain the images to plot as 1D vectors.
			n_rows : int, default=None
				number of rows within the plot. If None, the value is computed automatically.
			n_cols : int, default=None
				number of columns within the plot. If None, the value is computed automatically.
		"""
		if self.homogeneous_arrays is True:
			fig = plt.figure()

			# computes the number of rows and columns if not provided
			if n_cols is None:
				n_cols = int(np.sqrt(len(images)))
			if n_rows is None:
				n_rows = len(images)//n_cols

			axs = fig.subplots(n_rows, n_cols)

			if len(np.shape(axs)) == 1:
				axs = axs[:, None]

			# plot each image
			for i in range(n_rows):
				for j in range(n_cols):
					# compute the image index within the array
					k = i + j*n_cols

					# plot
					axs[i][j].imshow(images[k].reshape(self.n_px_v[0], self.n_px_h[0]), cmap="gray")
					axs[i][j].tick_params(axis='both', which="both", bottom=False, top=False, right=False, left=False, labelbottom=False, labeltop=False, labelright=False, labelleft=False)

			plt.subplots_adjust(left=0, right=1, wspace=0.05, hspace=0.05)
		else:
			raise NotImplementedError("Image plotting for non-homogeneous arrays is not implemented yet.")

	def __str__(self):
		str_ = f"Facial recognition database class ({self.n_persons} faces stored / {len(self.faces)} images)"
		return str_ 



# ----------------------------------
# 			  Function
# ----------------------------------

def search(image, catalog, plot=False):
	""" Returns the reference image with the minimum distance from the test image.

	This function returns the index of the image within the reference catalog which 
	distance (L2) squared to the test image is closest.

	This function can in fact be used with arrays other than actual images (e.g. Principal
	components, ...)

	Parameters
	----------
		image : numpy.ndarray (N)
			Test image.
		catalog : numpy.ndarray (M, N)
			Image reference catalog to which the test image is compared.
		plot : bool, default=False
			If True, the results of the search process will be computed

	Returns
	-------
		most_probable_id : int
			Index of the image (within the catalog) closest to the test image.
		cost_function : np.ndarray (M)
			Distances of each image of the catalog to the test image
	"""
	# distance squared between the test image and each image of the catalog
	cost_function = np.sqrt(np.sum((catalog - image)**2, axis=1))

	# id of the image closest to the test image
	most_probable_id = np.argmin(cost_function)

	# plot cost function and image found.
	if plot is True:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(cost_function, "-", color="darkblue")
		ax.plot(most_probable_id, cost_function[most_probable_id], "+", color="darkred", label="Result")

		ax.set_xlabel("$N$ [$-$]")
		ax.set_ylabel(r"$C(N) = \| X(N) - X_{ref}\|^2$")
		ax.legend()
	
	return most_probable_id, cost_function

def compute_confidence_matrix(reference_database, prediction, reference, criterion, threshold):
	""" Evaluates the performances of the recognition algorithm.

	Parameters
	----------
		reference_database : numpy.ndarray
			Master reference database.
		prediction : int, default=None
			Person indices found by the algorithm.
		reference : int, default=None
			Reference person indices.
		criterion : int, default=None
			Identification function.
		threshold : int, default=None
			Threshold value used for the automatic identification.
	"""
	# compute performance indices
	mask_found = prediction == reference
	n_false_positives 	= len(np.where(np.logical_and(criterion < threshold, np.invert(mask_found)))[0])
	n_true_positive 	= len(np.where(np.logical_and(criterion < threshold, mask_found))[0])
	n_false_negatives 	= len(np.where(np.logical_and(criterion > threshold, mask_found))[0])
	n_true_negative 	= len(np.where(np.logical_and(criterion > threshold, np.invert(mask_found)))[0])

	# print performances
	print("\nAlgorithm performances : ")
	print(f"    Identification rate : {n_true_positive/len(prediction)*100:.1f} %")
	print(f"    False positive rate : {n_false_positives/(n_false_positives + n_true_negative)*100:.1f} %")
	print(f"    False negative rate : {n_false_negatives/(n_false_negatives + n_true_positive)*100:.1f} %")

	performance_matrix = np.array([[n_true_positive, n_false_positives], [n_false_negatives, n_true_negative]])
	
	print("\nIdenfification performance matrix : ")
	print(performance_matrix)