# soft_2D_classical_dimers
 Monte Carlo (MC) simulation of 2-particle cluster phase of softly repulsive particles in the canonical ensemble.
 An overview is provided in `overview.ipynb` file.
 
 Prerequisites: MPI-C/C++ compiler,  Python

1) `jupyter-notebook Config_Creator_MC.ipynb`
(to generate initial config and input.lattice.dat)

2) edit `input.dat` either manually or exploiting/modifying `multiple_job.m100.py` script
(`multiple_job.m100.py` is intended for parallel submission of independent MC simulations on supercomputer assemblies run by SLURM scheduler but can be modified at ease). In the `input_scripts` folder you can also find a list of optimized parameters for the MC moves range.

3) compile with `make`

4) run by simply executing:  `mpirun -np R ./Monte_Carlo_NVT_MPI.x `

Note: the code is inherently parallel via simple MPI functions. Take care to use a suitable number of ranks (i.e. number of ranks R must be an integer divisor of the total number of steps in each block)

-----------------------------------------------

Parallel Tempering (PT) code works in a very similar way.

1) To set up: read and use the Jupyter Notebooks contained in the `input_scripts` folder

2) compile with `make`

3) To run:  `mpirun -np R ./Parallel_Tempering.x ` where `R`is also the number of temperatures simulated

-----------------------------------------------

We would like to cite the work by Percus and Kalos (O.E. Percus and M.H. Kalos, "Random Number Generators for MIMD Parallel Processors",  Journal of Parallel and Distributed Computing 6, 477-497 (1989)) for the algorithm underlying the random numbers generator (RNG) that we have implemented. In particular, here we include in MC and PT codes both the RANdom New York University (RANNYU) routine, in our custom C++ implementation. The original algorithm from which we have taken inspiration from was suggested by Knuth (see "Art of scientific computing"), and then adapted and improved by the group of M.H.Kalos. The RNG is a linear congruential which generates a sequence of random numbers which is different for each parallel process.
