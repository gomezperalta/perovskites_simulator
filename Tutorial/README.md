# perovskites_simulator
Given a compound, you can explore the probability it has to crystallize as an aristotype perovskite. The simulation is done with an Artificial Neural Network trained with the code patolli.py

The code of this repository that uses the trained Artificial Neural Network is simulate_compounds.py. 
You only need to type in the shell the next instruction:

$python simulate_compounds.py

When you run the code, this will ask you for the name of the txt-file containing the compounds to simulate. By default, the name is  compounds2simulate. 

In fact, you can simulate as many compounds as they are enlisted in compounds2simulate. Check the structure of that txt-file to  see the instructions given to simulate_compound.py. 

You can simulate a pure compound, a compound with vacancies or a solid solution. You need to specify which atoms are in the octahedral, cubeoctahedral and vertex sites of the aristotype perovskite structure. This is done in the entries  of compounds2simulate.txt "octahedron_atom", "cubeoctahedron_atom" and "vertex_atom". Furthermore, you also have to indicate the occupation fraction of the atoms in each site (entries "octahedron_frac", "cubeoctahedron_frac" and "vertex_frac").

The lattice parameter is proposed as the sum of the atoms in the octahedral and vertex sites. Other lattice parameters are considered within a deviation range of the proposed sum. The other lattice parameter are considered in the entries "maxdev" and "stepsize" of compounds2simulate.txt.

Besides compounds2simulate.txt, simulate_compounds.py requires the next files to work:

<ul>
  <li>neighdist.py, this file computes the local functions need as input data for the ANN.</li>
  <li>dictionary_upto_4Wyckoffsites.npy, this file contains the data needed to standardize the input data for the ANN.</li>
  <li>patolli_upto_4Wyckoffsites.h5, this file contains the ANN.</li>
</ul>

You should not remove the last three files if you don't want the program fails.

For each simulated compound, you will have as output files:
<ul>
  <li>A csv-file, which has the columns lattice parameter and the probability computed by the ANN to crystallize as an aristotype perovskite</li>
  <li>A png-file, with a plot of the csv-file</li>
 </ul>

You can check more details in the next paper https://doi.org/10.1016/j.jssc.2020.121253 . I also thank you in advance for your citation if you benefit of this code:

Juan I. Gómez-Peralta, Xim Bokhimi, J. Solid State Chemistry, Vol. 285, 121253 (2020)
