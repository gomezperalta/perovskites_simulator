# perovskites_simulator
Given a compound, you can explore the probability it has to crystallize as an aristotype perovskite. The simulation is done with an Artificial Neural Network trained with the code patolli.py

The code that uses the trained Artificial Neural Network is simulate_compounds.py
You only need to type in the shell the next instruction

$python simulate_compounds.py

When you run the code, this will ask you for the name of the txt-file containing the compounds to simulate. By default, the name is  compounds2simulate. 

In fact, you can simulate as many compounds as they are enlisted in compounds2simulate. Check the structure of that txt-file to to see the instructions given to simulate_compound.py.

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
