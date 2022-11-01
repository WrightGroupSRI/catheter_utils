import numpy as np

def get_coords(data_file):
    '''takes a path to a txt file and extracts coordinate data.
	   
	   data_file is the path to a .txt file that contains the 
	   x,y,z coordinates 
    '''

    file = open(data_file, 'r')
    lines = file.readlines()
    coords = []
    
    for line in lines: 
        data = line.split(' ')
        x, y, z = float(data[0]), float(data[1]), float(data[2])
        coords.append([x, y, z])
    
    return np.array(coords) 


def Chebyshev(distal_file, proximal_file, gt_coords, Geometry):
	'''produces the 95% chebyshev radius.
	
	   distal_file: text file of distal coordinates 
	   proximal_file: text file of proximal coordinates
	   gt_coords: ground truth tip coordinates 
	   Geometry: the geometry of the coils 
	'''
    
    distal_coords = get_coords(distal_file)
    proximal_coords = get_coords(proximal_file)
    
    fit_results = Geometry.fit_from_coils_mse(distal_coords, proximal_coords)
    tip_coords = (fit_results.tip).T

    Cov_mat = np.cov(tip_coords)
    Evals, Evects = np.linalg.eig(Cov_mat)
    
    sigma = max(Evals)
    return sigma * 2 * np.sqrt(5/3) 


def Bias(distal_file, proximal_file, gt_coords, Geometry): 
    '''produces the bias between the ground truth and measurements.
	
	   distal_file: text file of distal coordinates 
	   proximal_file: text file of proximal coordinates
	   gt_coords: ground truth tip coordinates 
	   Geometry: the geometry of the coils 
	'''
	
    distal_coords = get_coords(distal_file)
    proximal_coords = get_coords(proximal_file)
    
    fit_results = Geometry.fit_from_coils_mse(distal_coords, proximal_coords)
    mean_tip = np.mean(fit_results.tip, axis=0)
    
    bias_vect = mean_tip - gt_coords
    bias = np.linalg.norm(bias_vect)
    
    return bias