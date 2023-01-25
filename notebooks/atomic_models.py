#Adapted from Dr. Frédéric Poitevin on CompSPI

"""Read and write atomic models in various formats."""
import os

import gemmi
import numpy as np


def read_atomic_model(path, i_model=0, clean=True, assemble=True):
    """Read PDB or mmCIF file.
    Use Gemmi library to read PDB or mmCIF files and return a Gemmi model.
    The hierarchy in Gemmi follows:
    Structure - Model - Chain - Residue - Atom
    Parameters
    ----------
    path : string
        Path to PDB or mmCIF file.
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
    model: Gemmi Class
        Gemmi model
    Example
    -------
    TO DO
    Reference
    ---------
    See https://gemmi.readthedocs.io/en/latest/mol.html for a definition of
    gemmi objects.
    """
    if os.path.isfile(path):
        is_pdb = path.lower().endswith(".pdb")
        is_cif = path.lower().endswith(".cif")
        if is_pdb:
            model = _read_atomic_model_from_pdb(path, i_model, clean, assemble)
        elif is_cif:
            model = _read_atomic_model_from_cif(path, i_model, clean, assemble)
        else:
            model = None
            raise ValueError("File format not recognized.")
    else:
        model = None
        raise OSError("File could not be found.")
    return model
def _read_atomic_model_from_pdb(path, i_model=0, clean=True, assemble=True):
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
        Gemmi model
    """
    structure = gemmi.read_structure(path)
    if clean:
        structure = clean_gemmi_structure(structure)
    model = structure[i_model]
    if assemble:
        assembly = structure.assemblies[i_model]
        chain_naming = gemmi.HowToNameCopiedChain.AddNumber
        model = gemmi.make_assembly(assembly, model, chain_naming)
    return model
def _read_atomic_model_from_cif(path, i_model=0, clean=True, assemble=True):
    """Read Gemmi Model from CIF file.
    Parameters
    ----------
    path : string
        Path to mmCIF file.
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
        Gemmi model
    """
    cif_block = gemmi.cif.read(path)[0]
    structure = gemmi.make_structure_from_block(cif_block)
    if clean:
        structure = clean_gemmi_structure(structure)
    model = structure[i_model]
    if assemble:
        assembly = structure.assemblies[i_model]
        chain_naming = gemmi.HowToNameCopiedChain.AddNumber
        model = gemmi.make_assembly(assembly, model, chain_naming)
    return model
def clean_gemmi_structure(structure=None):
    """Clean Gemmi Structure.
    Parameters
    ----------
    structure : Gemmi Class
        Gemmi Structure object
    Returns
    -------
    structure : Gemmi Class
        Same object, cleaned up of unnecessary atoms.
    """
    if structure is not None:
        structure.remove_alternative_conformations()
        structure.remove_hydrogens()
        structure.remove_waters()
        structure.remove_ligands_and_waters()
        structure.remove_empty_chains()
    return structure
def write_atomic_model(path, model=gemmi.Model("model")):
    """Write Gemmi model to PDB or mmCIF file.
    Use Gemmi library to write an atomic model to file.
    Parameters
    ----------
    path : string
        Path to PDB or mmCIF file.
    model : Gemmi Class
        Optional, default: gemmi.Model()
        Gemmi model
    Reference
    ---------
    See https://gemmi.readthedocs.io/en/latest/mol.html for a definition of
    gemmi objects.
    """
    is_pdb = path.lower().endswith(".pdb")
    is_cif = path.lower().endswith(".cif")
    if not (is_pdb or is_cif):
        raise ValueError("File format not recognized.")
    structure = gemmi.Structure()
    structure.add_model(model, pos=-1)
    structure.renumber_models()
    if is_cif:
        structure.make_mmcif_document().write_file(path)
    if is_pdb:
        structure.write_pdb(path)


def write_cartesian_coordinates(path,chain_list):
    """Write Numpy array of cartesian coordinates to PDB or mmCIF file.
    Parameters
    ----------
    path : string
        Path to PDB or mmCIF file
    chain_list: list
        (chain name, list of atom types, list of arrays of atomic coordinates)
        ex.: (("A","N",([X1,Y1,Z1],...,[Xn,Yn,Zn])),("B",("N","CA","O"),([X1,Y1,Z1],...,[Xn,Yn,Zn])))
    -------
    """
    is_pdb = path.lower().endswith(".pdb")
    is_cif = path.lower().endswith(".cif")
    if not (is_pdb or is_cif):
        raise ValueError("File format not recognized.")

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
            atom.pos = gemmi.Position(
                iat[0],
                iat[1],
                iat[2],
            )
            atom.name = atom_types[atom_numb]
            structure[0][chain_name][0].add_atom(atom)
            atom_numb += 1

    if is_cif:
        structure.make_mmcif_document().write_file(path)
    if is_pdb:
        structure.write_pdb(path)