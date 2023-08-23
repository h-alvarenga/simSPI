from mpl_toolkits.mplot3d import axes3d
import numpy as np
import gemmi
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from simSPI.mpmodel.detergent_belt import Model


#Test for GLUT4 --------------------------------------------------------

#Create detergent belt
C = db.BeltCore()
C.set_belt_parameters(30, 20, 2)
C.set_atomic_parameters(1.8, "C")
C.generate_ellipsoid()

S = db.BeltShell()
S.set_belt_parameters(40, 30, 2)
S.set_atomic_parameters(1.5, "N")
S.generate_ellipsoid()

#Import and center protein
glut = db.MembraneProtein("data/pdb_files/7wsn.pdb")
glut.rotate_protein()

#Convex hull
C.in_hull(glut.final_coordinates)
C.remove()
 
S.in_hull(C.coordinates_set)
S.remove()
S.in_hull(glut.final_coordinates)      
S.remove()

#Export model to pdb file
file_name = "test_GLUT.pdb"
M = db.Model()
M.clean_gemmi_structure()
M.write_atomic_model(file_name, model=gemmi.Model("model"))
M.create_model(file_name,C,S,glut)


#Test for AQP0  --------------------------------------------------------

#Create detergent belt
C = db.BeltCore()
C.set_belt_parameters(35, 25, 2)
C.set_atomic_parameters(1.5, "C")
C.generate_ellipsoid()

S = db.BeltShell()
S.set_belt_parameters(40, 30, 2)
S.set_atomic_parameters(1.5, "N")
S.generate_ellipsoid()

#Import and center protein
aquaporin = db.MembraneProtein("data/pdb_files/2b6p.pdb")
aquaporin.rotate_protein(axis='X')

#Convex hull
C.in_hull(aquaporin.final_coordinates)
C.remove()
 
S.in_hull(C.coordinates_set)
S.remove()
S.in_hull(aquaporin.final_coordinates)      
S.remove()

#Export model to pdb file
file_name = "test_AQP.pdb"
M = db.Model()
M.clean_gemmi_structure()
M.write_atomic_model(file_name, model=gemmi.Model("model"))
M.create_model(file_name,C,S,aquaporin)
