import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import gemmi
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

"""Module that creates a model of a membrane protein surronded by a detergent belt"""

class Model():
    """Class to generate a detergent belt around a protein.
    """

    def __init__(self): pass

    def clean_gemmi_structure(self, structure=None):
        """Clean Gemmi Structure.
        
        Parameters
        ----------
        structure : Gemmi Class
            Gemmi Structure object.

        Returns
        -------
        structure : Gemmi Class
            Same object, cleaned up of unnecessary atoms.
            
        Reference
        ---------
        See https://github.com/compSPI/ioSPI/blob/master/ioSPI/atomic_models.py
        """
        if structure is not None:
            structure.remove_alternative_conformations()
            structure.remove_hydrogens()
            structure.remove_waters()
            structure.remove_ligands_and_waters()
            structure.remove_empty_chains()
        return structure

    def read_atomic_model_from_pdb(self, path, i_model=0, clean=True, assemble=True):
        """Read Gemmi Model from PDB file.
        
        Parameters
        ----------
        path : string
            Path to PDB file.
        i_model : integer
            Optional, default: 0
            Index of the returned model in the Gemmi Structure.
        clean : bool
            Optional, default: True
            If True, use Gemmi remove_* methods to clean up structure.
        assemble: bool
            Optional, default: True
            If True, use Gemmi make_assembly to build biological object.
        
        Returns
        -------
        model : Gemmi Class
            Gemmi Model
            
        Reference
        ---------
        See https://github.com/compSPI/ioSPI/blob/master/ioSPI/atomic_models.py
        """
        structure = gemmi.read_structure(path)
        if clean:
            structure = self.clean_gemmi_structure(structure)
            model = structure[i_model]
        if assemble:
            assembly = structure.assemblies[i_model]
            chain_naming = gemmi.HowToNameCopiedChain.AddNumber
            model = gemmi.make_assembly(assembly, model, chain_naming)
        return model
    
    def write_atomic_model(self, path, model=gemmi.Model("model")):
        """Write Gemmi model to PDB or mmCIF file.
        Use Gemmi library to write an atomic model to file.

        Parameters
        ----------
        path : string
            Path to PDB file.
        model : Gemmi Class
            Optional, default: gemmi.Model()
            Gemmi model

        Reference
        ---------
        See https://github.com/compSPI/ioSPI/blob/master/ioSPI/atomic_models.py
        See https://gemmi.readthedocs.io/en/latest/mol.html for a definition of
        gemmi objects.
        """
        structure = gemmi.Structure()
        structure.add_model(model, pos=-1)
        structure.renumber_models()
        structure.write_pdb(path)

    def write_cartesian_coordinates(self,path,chain_list):
        """Write Numpy array of cartesian coordinates to PDB or mmCIF file.
        
        Parameters
        ----------
        path : string
            Path to PDB file
        chain_list: list
            (chain name, list of atom types, atomic coordinates as numpy arrays)
            ex.: (("A","N",([X1,Y1,Z1],...,[Xn,Yn,Zn])),
                    ("B",("N","CA","O"),([X1,Y1,Z1],...,[Xn,Yn,Zn])))
                
        Reference
        ---------
        See https://github.com/compSPI/ioSPI/blob/master/ioSPI/atomic_models.py
        """
        structure = gemmi.Structure()
        structure.add_model(gemmi.Model("model"))
        structure.renumber_models()

        for chain in chain_list:
            chain_name = chain[0]
            atom_types = chain[1]
            atom_coord = chain[2]
            atom_numb = 0
            structure[0].add_chain(chain_name)
            structure[0][chain_name].add_residue(gemmi.Residue())
            for iat in atom_coord:
                atom = gemmi.Atom()
                atom.pos = gemmi.Position(iat[0],iat[1],iat[2])
                atom.name = atom_types[atom_numb]
                structure[0][chain_name][0].add_atom(atom)
                atom_numb += 1
        structure.write_pdb(path)

    def create_model(self,file_name,core,shell,protein=None):
        """This function generates a PDB file with the corona coordinates
        
        Parameters
        ----------
        file_name: string
            Path to PDB file.
        """
        pseudoatom1 = len(core.coordinates_set)*[core.atom_type]
        pseudoatom2 = len(shell.coordinates_set)*[shell.atom_type]
        
        if protein is not None:
            prtn_atoms = protein.atoms
            self.write_cartesian_coordinates(file_name,
                                         (("A",pseudoatom1,core.coordinates_set),
                                          ("B",pseudoatom2,shell.coordinates_set),
                                          ("C",prtn_atoms,protein.final_coordinates)))
        else:
            self.write_cartesian_coordinates(file_name,
                                         (("A",pseudoatom1,core.coordinates_set),
                                          ("B",pseudoatom2,shell.coordinates_set)))
                                         

class MembraneProtein(Model):
    """Class to import a protein in the proper format for the model"""
    
    def __init__(self,file_path):
        """This function sets the path for the protein pdb
        
        Prameters
        ---------
        file_path: string
            Path for pdb file.
        """
        is_pdb = file_path.lower().endswith(".pdb")
        if not is_pdb:
            raise ValueError("File format not recognized.")
        self.path = file_path
            
    def get_protein(self):
        """This function gets the X,Y and Z coordinates of a clean protein structure 
        (no alternative conformations, hydrogens, waters, ligands or empty chains)

        Returns
        -------
        X,Y,Z: numpy array[,3]
            A set of all atomic coordinates.
        """
        coord_set = []
        atoms_list = []
        model = self.read_atomic_model_from_pdb(self.path, clean=True)
        for chain in model:
            for res in chain:
                for atom in res:
                    atoms_list.append(atom.name)
                    coord = atom.pos
                    coord_set += ([[coord[0],coord[1],coord[2]]])
        self.atoms = atoms_list
        return np.array(coord_set)

    def center_protein(self):
        """This function centers a set of coordinates XYZ on the origin (0,0,0)
            
        Returns
        -------
        X,Y,Z: numpy array[,3]
            A set of all atomic coordinates centered on the origin.
        """
        coord_set = self.get_protein()
        mean_coord=[0,0,0]
        N = len(coord_set)
        for i in range(3):
            mean_coord[i] = sum(coord_set[:,i])/N
        new_coord_set = np.zeros((N,3))
        for point_number in range(N):
            for j in range(3):
                new_coord_set[point_number][j] = coord_set[point_number][j] - mean_coord[j]
        return np.array(new_coord_set)


    def rotate_protein(self, axis = 'Z'):
        """This function rotates the protein to an axis
            
        Parameters
        ----------
        axis: numpy array[,3]
            Vector to which the protein will be aligned.
            Optional, default: 'Z'
        
        Returns
        -------
        self.final_coordinates: numpy array[,3]
            Set of coordinates of rotated protein.
            
        Reference
        ---------
        See https://stackoverflow.com/questions/45142959/
        calculate-rotation-matrix-to-align-two-vectors-in-3d-space
        """
        if axis == 'X':
            axis = [1,0,0]
        elif axis == 'Y':
            axis = [0,1,0]
        elif axis == 'Z':
            axis = [0,0,1]
        else:
            raise ValueError("Axis not accepted.")
        
        def rotation_matrix(vec1,vec2):
            """This function finds the matrix to align two vectors
            
            Parameters
            ----------
            vec1: numpy array[,3]
                Vector to be aligned.
            vec1: numpy array[,3]
                Reference vector.
                
            Returns
            -------
            rotation matrix: numpy array[,3]
                Matrix to rotate vec1 to vec2. 
            """
            a = (vec1 / np.linalg.norm(vec1)).reshape(3)
            b = (vec2 / np.linalg.norm(vec2)).reshape(3)
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1-c) / (s**2))
            return rotation_matrix
        
        centered_protein = self.center_protein()
        pca = PCA(n_components=3)
        pca.fit(centered_protein)

        vec = pca.components_[0]
        mat = rotation_matrix(vec, axis)

        r = []
        for item in centered_protein:
            vec_rot = mat.dot(item)
            r.append([vec_rot[0],vec_rot[1],vec_rot[2]])
        self.final_coordinates = r

class DetergentBelt(Model):
    """Class to generate a detergent belt
    """

    def set_belt_parameters(self, axis1, axis2, height, center, thickness):
        """This function defines the micelle parameters according to user input

        Parameters
        ----------
        hight: float
            Height of ellipsoid.
        width: float
            Width of ellipsoid.
        e: float
            Ellipticity.
        center: numpy array[,3]
            Center of ellipsoid.
        thickness: float
            Thickness of micelle shell.
                                                                                                           
        Returns
        -------
        a: float
            Major axis of ellipsoid.
        b: float
            Minor axis of ellipsoid.
        c: float
            Minor axis of ellipsoid.
        """
        a = axis1
        b = axis2
        c = height

        self.parameters = (a,b,c)
        self.center = center
        self.thickness
        
    def set_core_atomic_parameters(self, r, atom_type):
        """This function defines the pseudo atoms parameters according to user input

        Parameters
        ----------
        r: float 
            Atomic ray.
        atom_type: string
            Atomic element.
        """

        self.core_atom_type = atom_type
        self.core_atomic_ray = r
        
    def set_shell_atomic_parameters(self, r, atom_type):
        """This function defines the pseudo atoms parameters according to user input

        Parameters
        ----------
        r: float 
            Atomic ray.
        atom_type: string
            Atomic element.
        """

        self.shell_atom_type = atom_type
        self.shell_atomic_ray = r
        
    def eq_ellipsoid(self,x,y,z,a,b,c):
        """This function checks if a point [x,y,z] is inside an ellipsoid of axis b and c

        Parameters
        ----------
        x: float
            X coordinate.
        y: float
            Y coordinate.
        z: float
            Z coordinate.
        a: float
            Major axis of ellipsoid.
        b: float
            Major axis of ellipsoid.
        c: float
            Minor axis of ellipsoid.

        Returns
        -------
        eq <= 1: bool
            True if point belongs to the ellipsoid
            False if the point is not located in the ellipsoid
        """
        eq = ((x**2)/a**2) + ((y**2)/b**2) + ((z**2)/c**2)
        return eq <= 1
    
    def generate_core_ellipsoid(self):
        """This function creates an ellipsoid by generating equally spaced points
        from the origin (0,0,0) until the ellipsoid limits are reached.
        
        Returns
        -------
        self.coordinates_set: numpy array[,3]
            Set of coordinates of ellipsoid pseudo atoms.
        """
        coordinates_set = []        
        x0 = self.center[0]
        y0 = self.center[1]
        z0 = self.center[2]
        t = self.thickness
        a,b,c = (self.parameters+t)
        r = self.core_atomic_ray
        z = r
        while z <= c:
            y = r
            while y <= b:
                x = r
                while x <= a:
                    if self.eq_ellipsoid(x,y,z,a,b,c):
                         coordinates_set += ([x+x0,y+y0,z+z0], [-x+x0,y+y0,z+z0], 
                                             [x+x0,-y+y0,z+z0], [-x+x0,-y+y0,z+z0],
                                             [x+x0,y+y0,-z+z0], [-x+x0,y+y0,-z+z0], 
                                             [x+x0,-y+y0,-z+z0], [-x+x0,-y+y0,-z+z0])
                    x += 2*r
                y += 2*r
            z += 2*r
        self.coordinates_set = coordinates_set
        
    def generate_shell_ellipsoid(self):
        """This function creates an ellipsoid by generating equally spaced points
        from the origin (0,0,0) until the ellipsoid limits are reached.
        
        Returns
        -------
        self.coordinates_set: numpy array[,3]
            Set of coordinates of ellipsoid pseudo atoms.
        """
        coordinates_set = []        
        x0 = self.center[0]
        y0 = self.center[1]
        z0 = self.center[2]
        a,b,c = self.parameters
        r = self.atomic_ray
        z = r
        while z <= c:
            y = r
            while y <= b:
                x = r
                while x <= a:
                    if self.eq_ellipsoid(x,y,z,a,b,c):
                         coordinates_set += ([x+x0,y+y0,z+z0], [-x+x0,y+y0,z+z0], 
                                             [x+x0,-y+y0,z+z0], [-x+x0,-y+y0,z+z0],
                                             [x+x0,y+y0,-z+z0], [-x+x0,y+y0,-z+z0], 
                                             [x+x0,-y+y0,-z+z0], [-x+x0,-y+y0,-z+z0])
                    x += 2*r
                y += 2*r
            z += 2*r
        self.coordinates_set = coordinates_set

    def in_hull(self,hull_set):
        """This function list all points form a set that are inside a convex hull

        Parameters
        ----------
        hull_set: numpy array[,3]
            A set of atomic coordinates.

        Reference
        ---------
        See https://stackoverflow.com/questions/29311682/finding-if-point-is-in-3d-poly-in-python
        """
        hull = ConvexHull(hull_set)
        inside = list()
        for point in self.coordinates_set:
            new_hull = ConvexHull(np.concatenate((hull_set, [point]),axis=0))
            if np.array_equal(new_hull.vertices, hull.vertices):
                inside.append(point)
        self.inside = inside

    def remove(self):
        """This function removes the points of the ellipsoid coordinates set
        that are inside the hull
        """
        for point in self.inside:
            self.coordinates_set.remove(point)


class BeltCore(DetergentBelt):
        pass

class BeltShell(DetergentBelt):
        pass
