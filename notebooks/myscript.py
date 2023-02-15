from simSPI import detergent_belt as mpm

# for profiling/bechmarking
def main():
  aqp = mpm.MembraneProtein("2b6p.pdb") #path to pdb file
  aqp.rotate_protein(axis='X')

  C = mpm.BeltCore()
  C.set_belt_parameters(35, 25, 2)    #set the parameters for the core of your micelle
  C.set_atomic_parameters(1.5, "CA")  #set the parameters for you hydrophilic pseudoatom
  C.generate_ellipsoid()

  S = mpm.BeltShell()
  S.set_belt_parameters(40, 30, 2)    #set the parameters for the shell of your micelle
  S.set_atomic_parameters(1.5, "N")   #set the parameters for you hydrophobic pseudoatom
  S.generate_ellipsoid()

  C.in_hull(aqp.final_coordinates)
  C.remove()

if __name__ == '__main__':
  main()