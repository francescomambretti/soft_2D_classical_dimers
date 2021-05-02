/****************************************************************
*****************************************************************
    _/      _/_/_/  _/_/_/  Laboratorio di Calcolo Parallelo e
   _/      _/      _/  _/  di Simulazioni di Materia Condensata
  _/      _/      _/_/_/  c/o Sezione Struttura della Materia
 _/      _/      _/      Dipartimento di Fisica
_/_/_/  _/_/_/  _/      Universita' degli Studi di Milano
*****************************************************************
*****************************************************************/

/* 02/05/2021

Code version: 9.0 (internal code development of the group)

Contributions by: Fabio Civillini, Matteo Martinelli, Francesco Mambretti, Davide E. Galli

MPI code: parallel Monte Carlo simulation for the equilibrium phase of 2D soft matter in the n=2 cluster phase

- There is an equilibration phase and the possibility to restart from a previous configuration & thermodynamic state, by also saving data blocking statistics already performed
- Features single particle and cluster moves both
- Can simulate both a perfect triangular lattice and a deformed one
        
*/
// GITHUB VERSION

#include"Monte_Carlo_NVT_MPI.h"
#include"Random.h"

int main(int argc, char** argv)
{ 
	MPI_Init(&argc, &argv);

//measure initial time for performance
	double start, end;
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

//Inizialization		
	Input();
  	int nstep_per_rank=nstep/size; // each rank will process nstep_per_rank steps within each block
  	if (my_rank==rank_0)
  		{
  		if((nstep%size)!=0)
  			{
			cerr << "Number of ranks chosen is not admitted. Please change number of ranks!" << endl;
			exit(EXIT_FAILURE);
			}
		}  	
  	  	
// Give each rank a different set of parameters for parallel random numbers generation
	ifstream Primes("Primes");
  	int p1,p2,count=0;
  	while(count<= my_rank) 
  	{
		Primes >> p1 >> p2 ;
		count++;  
	} 
 	Primes.close();

	SetRandom(seed,p1,p2);
	
	MPI_Barrier(MPI_COMM_WORLD);
	
    // Equilibration
	int iblk,istep,j;
	for(iblk=1; iblk <= nblkt; ++iblk)
  	{
		for(istep=1; istep <= nstep_per_rank; ++istep)
    	{
      	// Single particle moves
            for (j=0; j < 3; ++j)
                Move(j);
      	
      	// Dimer moves
      		Cluster();	
      		Move_Rotate();
      		Move_Break();
      		Move_Cluster();
    	}
		if (my_rank==rank_0)	cout << "Equilibration block "<< iblk << endl;
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

//Simulation  	
  	for(iblk=1+oblk; iblk <= nblk; ++iblk) //loop over blocks
  	{
    		Reset(iblk);   										//Reset block averages
    		if (my_rank==rank_0) Reset_mpi();					//Set to 0 Mpi_rank_0
    		for(istep=1; istep <= nstep_per_rank; ++istep)
    		{
            // Single particle moves
                for (j=0; j < 3; ++j)
                    Move(j);
                    
      		// Dimer moves
      			Cluster();
      			Move_Rotate();
      			Move_Break();
      			Move_Cluster();
      		
      			Measure();
      			Accumulate(); 									//Update block averages
    		}//end for istep

    	// Merge the data obtained by the different ranks
	   	MPI_Reduce(blk_av, mpi_blk_av, n_props, MPI_LONG_DOUBLE, MPI_SUM, rank_0, MPI_COMM_WORLD);
		MPI_Reduce(&blk_norm, &mpi_blk_norm, 1, MPI_DOUBLE, MPI_SUM, rank_0, MPI_COMM_WORLD); 

		MPI_Reduce(accepted,mpi_accepted, n_moves, MPI_INT, MPI_SUM, rank_0, MPI_COMM_WORLD);
		MPI_Reduce(attempted, mpi_attempted, n_moves, MPI_INT, MPI_SUM, rank_0, MPI_COMM_WORLD);
		
	   	MPI_Barrier(MPI_COMM_WORLD);
	   	if (my_rank==rank_0) Averages(iblk);
	}
  	
  	if (my_rank==rank_0) ConfFinal();

	delete [] walker;
	delete [] blk_av;
	delete [] glob_av;
	delete [] glob_av2;
	delete [] mpi_blk_av;
	delete [] particle;
	delete [] bin_counter_x;
    delete [] bin_counter_obl;

	//measure end time
	MPI_Barrier(MPI_COMM_WORLD);
	end=MPI_Wtime();
    
	if (my_rank==rank_0)
    	{
		cout << "nblk= " <<nblk<<"        nstep= " << nstep << "      npart= " << npart << endl;
        	cout << size << " time  " << end-start << endl;
	}
	
	MPI_Finalize();
    
	return 0;
}

//**********************************************************************************************************

//==========================================
// Functions
//==========================================

//==================
// Input
//==================

void Input(void)
{
	if (my_rank==rank_0) //only rank master
	{
	 	ifstream ReadSeed, file_input;
	  
	//Read input informations
		file_input.open("input.dat");

	//Restart check
		file_input >> restart;

	//Read seed for random numbers
		if (!restart) ReadSeed.open("seed");
		else ReadSeed.open("output.seed");
		ReadSeed >> seed[0] >> seed[1] >> seed[2] >> seed[3];
		ReadSeed.close();	  

		file_input >> temp;
		beta = 1.0/temp;
	
        //Read hard-core potential parameters
		file_input >> khc; //Hard-core
		file_input >> r_hc;
		r_hc2=r_hc*r_hc;
	

		if (restart) 		//Read parameters for data blocking and lattice after restart
		{
			ifstream ReadRestart;
			ReadRestart.open("restart.dat");
			ReadRestart >> oblk;

			ReadRestart >> u1x;
			ReadRestart >> u2x;	
			ReadRestart >> u2y;		
			ReadRestart >> nx;
			ReadRestart >> ny;
			ReadRestart >> ncluster;
			ReadRestart >> rho;
            ReadRestart >> angle;
            sinang = sin(angle);
            cosang = cos(angle);

			ReadRestart >> glob_epot;			ReadRestart >> glob_epot2;
			ReadRestart >> glob_heat;			ReadRestart >> glob_heat2;

			ReadRestart.close();
		}
		else 			// Read lattice parameters
		{
			ifstream ReadLattice;
			oblk = 0;
			ReadLattice.open("input.lattice.dat");
		
			ReadLattice >> u1x;
			ReadLattice >> u2x;
			ReadLattice >> u2y;
			ReadLattice >> nx; //number of elementary lattice cells
			ReadLattice >> ny; // idem
			ReadLattice >> ncluster; // here, always = 2
			ReadLattice >> rho; // density
            ReadLattice >> angle;
            sinang = sin(angle);
            cosang = cos(angle);
			
			ReadLattice.close();
		}

		npart=nx*ny*ncluster;
		nspin=npart/2; //ok for even number of particles

        //Normalize rectangular cell
		u2x /= u1x;
		u2y /= u1x;
		u1x = 1.0;
		tga = u2y/u2x;
		cout << "Size of normalized rectangular cell : " << u1x << "x" << u2y << endl;
		cout << "Third basis vector " << u2x << endl;
		
		int ncells = nx*ny;
		cout << "Number of cells = " << nx << "x" << ny << endl;
		double scell = u1x * u2y;  //area elementary cell normalized
		vol = (double)npart/rho;
		double acell = sqrt(vol/((double)(ncells)*scell)); // scale factor for physical units
		u1x *= acell;
		u2y *= acell;
		u2x *= acell;
		boxx = nx * u1x;
		boxy = ny * u2y;
		my_boxx = boxx-boxy/tanang;  // used for PBC when mapping the actual box onto the corresponding rectangular one. Here we use the projection of the top right corner on the x axis
			
	// Read cutoff radius for interactions and parameters for building Verlet neighbors lists
		file_input >> rcut;
        rcut2=rcut*rcut;
		file_input >> check_verlet; //boolean value
		file_input >> rverlet; // Verlet radius for neighbors lists
        rverlet=rverlet*rverlet;
					
	//Tail corrections for potential energy
		vtail = V_Tail();
	
	// Read moves parameters
		file_input >> delta[0];
		file_input >> delta[1];
		file_input >> delta[2];
		file_input >> alpha; // maximum rotation angle for dimers
		file_input >> deltaB; // break parameter
		file_input >> deltaCM; // delta for center of mass translation
	
	// Read blocks and steps
		file_input >> nblk;
		file_input >> nblkt; //equilibration
		file_input >> nstep; 

		nblk = nblk + oblk;  //sum old blocks (in case of restart)
	
	// Print initial conditions
		cout << "Monte Carlo simulation" << endl << endl;
		cout << "Triangular lattice with "<<ncluster<<" particles in each cluster"<<endl;
		cout << "Number of particles = " << npart << endl;
		cout << "Density of particles = " << rho << endl;
		cout << "Volume of the simulation box = " << vol << endl;
		cout << "Edges of the simulation box = (" << setprecision(10) << boxx << " ; " << boxy <<")"<< endl;

		cout << "Boltzmann weight exp(- beta * sum_{i<j} v(r_ij) ), beta = 1/T " << endl << endl;
		cout << "Temperature = " << temp << endl;
		cout << "Cutoff of the interatomic potential = " << rcut << endl << endl;
		cout << "Tail correction for the potential energy = " << vtail << endl;
		cout << "The program perform Metropolis moves with uniform translations" << endl;
		cout << "50% Moves parameter = " << delta[2] << endl;
		cout << "20% Moves parameter = " << delta[1] << endl;
		cout << "00% Moves parameter = " << delta[0] << endl;
		cout << "Rotation of alpha= "<< alpha	<< endl;
		cout << "Break of deltaB= "<< deltaB	<< endl;
		cout << "Traslation of cluster of deltaCM= "<< deltaCM	<< endl;
		cout << "Number of old blocks = " << oblk << endl;
		cout << "Number of new blocks = " << nblk  - oblk << endl;
		cout << "Number of thermalization blocks = " << nblkt << endl;
		cout << "Number of steps in one block = " << nstep << endl << endl;
		file_input.close();
	}

// Broadcast input params to all ranks
	MPI_Bcast(seed,4,MPI_INT,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&temp,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&beta,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	
	MPI_Bcast(&npart,1,MPI_INT,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&nspin,1,MPI_INT,rank_0,MPI_COMM_WORLD);
	
	MPI_Bcast(&khc,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&r_hc,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&r_hc2,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	
	MPI_Bcast(&oblk,1,MPI_INT,rank_0,MPI_COMM_WORLD);	
	MPI_Bcast(&nblk,1,MPI_INT,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&nblkt,1,MPI_INT,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&nstep,1,MPI_INT,rank_0,MPI_COMM_WORLD);
	
	MPI_Bcast(&nx,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);	
	MPI_Bcast(&ny,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&rho,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&vol,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&boxx,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&my_boxx,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&boxy,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&qminx,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&qminy,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&qmino,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&sinang,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&tanang,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&cosang,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&u1x,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&u2x,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&u2y,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&tga,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	
	MPI_Bcast(delta,n_delta,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&alpha,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&deltaB,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&deltaCM,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	
	MPI_Bcast(&rcut,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&rcut2,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&check_verlet,1,MPI_LOGICAL,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&rverlet,1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&vtail,1,MPI_LONG_DOUBLE,rank_0,MPI_COMM_WORLD);
	
	MPI_Barrier(MPI_COMM_WORLD);

    //Prepare arrays for measurements
	n_props = n_obs; //n_props contains the number of observables, to correctly size dynamic arrays

    //inizializzation of g(r)
	igofr = n_obs;
	igspin_x = igofr + nbins; // dimer-dimer correlation functions
    igspin_obl = igspin_x + nx/2; //ok for even and odd both
	if (boxx<boxy)	bin_size = (boxx/2.0)/(double)nbins;
	else 			bin_size = (boxy/2.0)/(double)nbins;
	  
    //inizializzation of S(q)
	qminx = 2.0*pi/boxx;
    qminy = 2.0*pi/boxy;
	qmino = 2.0*pi*sinang/boxy;

	nbinqx= 10*nx; 					// number of bins for S(q) along x,y and oblique axes
    nbinqy= 10*ny;
	nbinqo= 10*ny;


    //assign starting indexes for S(q) in the observables arrays
    isofqx = igspin_obl + ny/2; //ok for even and odd both
	isofqy = isofqx + nbinqx;
	isofqo = isofqy + nbinqy;
	isofq_clx = isofqo + nbinqo;
    isofq_cly = isofq_clx + nbinqx;
	isofq_clo = isofq_cly + nbinqy;
	n_props = isofq_clo + nbinqo;

    //Dynamic allocation of arrays
	walker= new long double[n_props]();
	blk_av=new long double[n_props]();
	glob_av=new long double[n_props]();
	glob_av2=new long double[n_props]();
	mpi_blk_av = new long double[n_props]();
	bin_counter_x = new int [nx/2]();
    bin_counter_obl = new int [ny/2]();

    // Save hare particles properties
	particle=new Particle[npart]();

    // Initialize arrays for specific heat and potential energy
	if (restart) 
	{	
		glob_av[iv] = glob_epot;		glob_av2[iv] = glob_epot2;
		glob_av[iv2] = glob_heat;		glob_av2[iv2] = glob_heat2;
		if (my_rank==rank_0) 
		{
			double xx, yy, ii;
			
			ifstream ReadConf;
			cout << "Read initial configuration from file output.config.final.txt " << endl << endl;
		  	ReadConf.open("output.config.final.txt");
		  	for (int i=0; i<npart; ++i)
				{
				ReadConf >> xx >> yy >> ii;	
				particle[i].x=Pbcx( xx * boxx , yy * boxy);
				particle[i].y=Pbcy( yy * boxy );
				particle[i].pointer=i; // the other particle in the dimer
				particle[i].i_cluster=-1; // initial value (meaningless)
				particle[i].xm=particle[i].x; // barycenter coordinates
				particle[i].ym=particle[i].y;
				particle[i].phi=0.;
				particle[i].dr=0.;
				}
		  	ReadConf.close();
		  	Check_Configuration(); // check that particles are found at a distance slightly larger than the hard core radius. If not, they are moved a little bit one away from each other
		}
	}
	else
	{
	//Read initial configuration
		if (my_rank==rank_0)
			{
			double xx, yy, ii;
			
			ifstream ReadConf;
			cout << "Read initial configuration from file config.0 " << endl << endl;
		  	ReadConf.open("config.0");
		  	for (int i=0; i<npart; ++i)
				{
				ReadConf >> xx >> yy >> ii;				 
				particle[i].x=Pbcx( xx * boxx, yy * boxy );
				particle[i].y=Pbcy( yy * boxy );
				particle[i].pointer=i;
				particle[i].i_cluster=-1;
				particle[i].xm=particle[i].x;
				particle[i].ym=particle[i].y;
				particle[i].phi=0.;
				particle[i].dr=0.;
				}
		  	ReadConf.close();
		  	Check_Configuration();
		}
	}
	
// Define MPI_PARTICLE -> new data type

//Variables for the single particle
	var_len[0]=1; 	var_add[0] = offsetof(Particle, x); 			var_type[0]=MPI_DOUBLE;
	var_len[1]=1; 	var_add[1] = offsetof(Particle, y); 			var_type[1]=MPI_DOUBLE;
//Variables for the particle pair
	var_len[2]=1; 	var_add[2] = offsetof(Particle, pointer); 	    var_type[2]=MPI_INTEGER;
	var_len[3]=1; 	var_add[3] = offsetof(Particle, i_cluster); 	var_type[3]=MPI_INTEGER;
//Spin variables
	var_len[4]=1; 	var_add[4] = offsetof(Particle, xm); 			var_type[4]=MPI_DOUBLE;
	var_len[5]=1; 	var_add[5] = offsetof(Particle, ym); 			var_type[5]=MPI_DOUBLE;
	var_len[6]=1; 	var_add[6] = offsetof(Particle, phi); 		    var_type[6]=MPI_DOUBLE;
	var_len[7]=1; 	var_add[7] = offsetof(Particle, dr); 			var_type[7]=MPI_DOUBLE;

	MPI_Type_create_struct(num_var, var_len, var_add, var_type, &MPI_PARTICLE);
	MPI_Type_commit(&MPI_PARTICLE);
	
	MPI_Bcast(particle,npart,MPI_PARTICLE,rank_0,MPI_COMM_WORLD);
	
	if (check_verlet)
        Def_neighbors(); //create Verlet lists

	if (my_rank==rank_0)
        Print_Parameters();
}

//==================
// Move
//==================

void Move(int acc_index) // single particle translation
{
	int o;
	long double p, energy_old, energy_new;
	double xnew, ynew;
	
	for(int i=0; i<npart; ++i) // try the move npart times
	{
	//Old
		o = (int)(Rannyu()*npart);
   		energy_old = Boltzmann(particle[o].x, particle[o].y, o); //evaluate the contribution to the total potential energy given by the particle o

	//New
		xnew = particle[o].x + delta[acc_index]*(Rannyu() - 0.5);
		ynew = particle[o].y + delta[acc_index]*(Rannyu() - 0.5);

		xnew = Pbcx (xnew, ynew);
		ynew = Pbcy (ynew);
		
		energy_new = Boltzmann(xnew, ynew, o); //evaluate the contribution to the total potential energy given by the particle o in the new proposed position

	//Metropolis test
		p = exp(beta*(energy_old-energy_new));
		if(p >= Rannyu())  
		{
		//Update	
			particle[o].x = xnew;
			particle[o].y = ynew;
		
			accepted[acc_index] += 1;
		}
		attempted[acc_index] += 1;
	}
}

//==================
// Move_Cluster
//==================

void Move_Cluster() // move a dimer
{
	int o, po;
	long double p, energy_old, energy_new;
	double x1_new, y1_new;
	double x2_new, y2_new;
	double phi, xm, ym, dr, cosPhi, senPhi;
	
	for(int i=0; i<nspin; ++i)	// when we move a spin, 2 particles are involved. try this move nspin times
	{
	//Select randomly a particle (for C++ syntax, 0 <= o <= npart-1)
		o = (int)(Rannyu()*npart);
   		po = particle[o].pointer;
   		
	//Old	
		energy_old = Boltzmann_2(particle[o].x, particle[o].y, o, particle[po].x, particle[po].y, po); //evaluate the contribution to the total potential energy given by the dimer

	//New
		xm = particle[o].xm+(Rannyu()-0.5)*deltaCM;
		ym = particle[o].ym+(Rannyu()-0.5)*deltaCM;
		xm = Pbcx (xm, ym);
		ym = Pbcy (ym);
		dr = particle[o].dr;
		phi= particle[o].phi;
	
		cosPhi=cos(phi)/2.0;		
		senPhi=sin(phi)/2.0;		
		x1_new=Pbcx(xm+dr*cosPhi, ym+dr*senPhi); 	x2_new=Pbcx(xm-dr*cosPhi, ym-dr*senPhi);	
		y1_new=Pbcy(ym+dr*senPhi);					y2_new=Pbcy(ym-dr*senPhi);
        
		energy_new = Boltzmann_2(x1_new,y1_new,o, x2_new, y2_new, po); //evaluate the contribution to the total potential energy given by the dimer in the new proposed position
		
        // Metropolis test
		p = exp(beta*(energy_old-energy_new));
		if(p >= Rannyu())  
		{
		//Update
			particle[o].x = x1_new;	particle[po].x = x2_new;
			particle[o].y = y1_new;	particle[po].y = y2_new;
			
			particle[o].xm=xm;		particle[po].xm=xm;
			particle[o].ym=ym;		particle[po].ym=ym;
		
			accepted[icm] += 1;
		
		// Check again
			Make_Spin(o);	
			Make_Spin(po);
		
		// If the particle pair is now broken (impossible at low temperature), re-build again all the clusters
			if (o!=particle[po].pointer || po!=particle[o].pointer) Cluster();
		}
		attempted[icm] += 1;
	}
		
}

//==================
// Move_Rotate
//==================

void Move_Rotate() //Rotate a dimer around its barycenter
{
	int o, po;
	long double p, energy_old, energy_new;
	double x1_new, y1_new;
	double x2_new, y2_new;
	double phi, xm, ym, dr, cosPhi, senPhi;

	for(int i=0; i<nspin; ++i)
	{
	//Select randomly a particle pair
		o = (int)(Rannyu()*npart);
   		po = particle[o].pointer;
   		
	//Old	
		energy_old = Boltzmann_2(particle[o].x, particle[o].y, o, particle[po].x, particle[po].y, po);

	//New
		xm = particle[o].xm;		
		ym = particle[o].ym;
		dr = particle[o].dr;
		phi=Pbca(alpha*(Rannyu()-0.5)+particle[o].phi);
		
		cosPhi=cos(phi)/2.0;
        senPhi=sin(phi)/2.0;
		
		x1_new=Pbcx(xm+dr*cosPhi, ym+dr*senPhi); 	x2_new=Pbcx(xm-dr*cosPhi, ym-dr*senPhi);	
		y1_new=Pbcy(ym+dr*senPhi);					y2_new=Pbcy(ym-dr*senPhi);
		energy_new = Boltzmann_2(x1_new,y1_new,o, x2_new, y2_new, po);

	//Metropolis test
		p = exp(beta*(energy_old-energy_new));
		if(p >= Rannyu())  
		{
		//Update
			particle[o].x = x1_new;	particle[po].x = x2_new;
			particle[o].y = y1_new;	particle[po].y = y2_new;
			particle[o].phi=phi;	particle[po].phi=phi;
		
			accepted[ir] += 1;
		
			Make_Spin(o);	
			Make_Spin(po);
		
			if (o!=particle[po].pointer || po!=particle[o].pointer) Cluster();
		}
		attempted[ir] += 1;
	}
}

//==================
// Move_Break
//==================

void Move_Break()
{
	int o, po;
	long double prob, energy_old, energy_new;
	double x1_new, y1_new;
	double x2_new, y2_new;
	double xm, ym, dr, cosPhi, senPhi;

	for(int i=0; i<nspin; ++i)	//quando muovo uno spin, muovo 2 particles (devo fare metà delle mosse)
	{
	//Select randomly a particle (for C++ syntax, 0 <= o <= npart-1)
		o = (int)(Rannyu()*npart);
   		po = particle[o].pointer;
   		
	//Old	
		energy_old = Boltzmann_2(particle[o].x, particle[o].y, o, particle[po].x, particle[po].y, po);

	//New
		xm = particle[o].xm;		
		ym = particle[o].ym;
		dr = particle[o].dr+(Rannyu()-0.5)*2.0*deltaB;					//dilato/collasso la distanza fra le particles - fattore 2.0 legato a r=d/2
			
		cosPhi=cos(particle[o].phi)/2.0;								//fattore 2.0 legato a r=d/2
		senPhi=sin(particle[o].phi)/2.0;	
		
		x1_new=Pbcx(xm+dr*cosPhi, ym+dr*senPhi); 	x2_new=Pbcx(xm-dr*cosPhi, ym-dr*senPhi);	
		y1_new=Pbcy(ym+dr*senPhi);					y2_new=Pbcy(ym-dr*senPhi);
		energy_new = Boltzmann_2(x1_new,y1_new,o, x2_new, y2_new, po);

	//Metropolis test
		prob = exp(beta*(energy_old-energy_new));
		if(prob >= Rannyu())  
		{
		//Update
			particle[o].x = x1_new;	particle[po].x = x2_new;
			particle[o].y = y1_new;	particle[po].y = y2_new;
			particle[o].dr=dr;		particle[po].dr=dr; //ok salvare i dr così perchè non sono quadratici
		
			accepted[ib] += 1;
		
		//Rivalutazione delle coppie di particles
			Make_Spin(o);	
			Make_Spin(po);
		
		//Se ho rotto la coppia, prova a ciclare su tutto	
			if (o!=particle[po].pointer || po!=particle[o].pointer) Cluster();
		}
		attempted[ib] += 1;
	}
}

//==================
// Boltzmann
//==================

long double Boltzmann(double xx, double yy, int ip) // returns the potential energy contribution involving particle ip to the total potential energy
{
	long double ene=0.0;
	double dx, dy, dR;
	int jpointer, i, maximum=neighbors[ip].size();

	if (check_verlet) // Evaluation through Verlet lists
	{
		for (i=0; i<maximum ; ++i)
		{
			jpointer=neighbors[ip][i];
			dx = Pbcx(xx - particle[jpointer].x, yy - particle[jpointer].y);
			dy = Pbcy(yy - particle[jpointer].y);
			dR = dx*dx + dy*dy;

			ene += V_Pot(dR);
		}
	}
	else	// Crosscheck ip with all the other particles
	{
		for (i=0; i<npart; ++i)
		{
			if(i != ip)
			{
			// distance ip-i in pbc
				dx = Pbcx(xx - particle[i].x, yy - particle[i].y);
				dy = Pbcy(yy - particle[i].y);
				dR = dx*dx + dy*dy;

				ene += V_Pot(dR);
			}
		}
	}
	
	return ene;
}

//==================
// Boltzmann_2
//==================

long double Boltzmann_2(double x1, double y1, int i1, double x2, double y2, int i2) // returns the potential energy contribution involving particles i1 and i2 (which form a dimer) to the total potential energy
{
	long double ene=0.0;
	double dx, dy, dr;
	int jpointer, maximum1, maximum2, i;

	if(check_verlet)	// Use Verlet
	{
        maximum1 = neighbors[i1].size();
        maximum2 = neighbors[i2].size();
	//Interazioni della prima particle (esclusa la seconda)
		for (i=0; i<maximum1; ++i)
		{
			jpointer=neighbors[i1][i];
			if (jpointer!=i2)
			{
				dx = Pbcx(x1 - particle[jpointer].x, y1 - particle[jpointer].y);
				dy = Pbcy(y1 - particle[jpointer].y);
				dr = dx*dx + dy*dy;

				ene += V_Pot(dr);
			}
		}

	//Interazioni della seconda particle (esclusa la prima)	
		for (i=0; i< maximum2; ++i)
		{
			jpointer=neighbors[i2][i];
			if (jpointer!=i1)
			{
				dx = Pbcx(x2 - particle[jpointer].x, y2 - particle[jpointer].y);
				dy = Pbcy(y2 - particle[jpointer].y);
				dr = dx*dx + dy*dy;

				ene += V_Pot(dr);
			}
		}
				
	}
	else	// Use all the particles
	{	
		for (i=0; i<npart; ++i)
		{
			if(i != i1 && i != i2)
			{
			// distance ip-i in pbc Particle i1
				dx = Pbcx(x1 - particle[i].x, y1 - particle[i].y);
				dy = Pbcy(y1 - particle[i].y);
				dr = dx*dx + dy*dy;
				
				ene += V_Pot(dr);
				
			// distance ip-i in pbc Particle i2
				dx = Pbcx(x2 - particle[i].x, y2 - particle[i].y);
				dy = Pbcy(y2 - particle[i].y);
				dr = dx*dx + dy*dy;
				
				ene += V_Pot(dr);	
			}
		}

	}
 
 // Internal interaction between Particle i1 and Particle i2
     dx = Pbcx(x2 - x1, y2 - y1);
     dy = Pbcy(y2 - y1);
     dr = dx*dx + dy*dy;

     ene += V_Pot(dr);
	
	return ene;
}

//==================
// Cluster
//==================

void Cluster()  // build dimers
{
	double xm, ym, dr, phi;
	int flag=0,i=0;
    
	while (flag!=npart)
	{		
		for (i=0; i<npart ; ++i)
            Make_Spin(i); //Check that all particles are coupled
        
		flag=0; //check that there are no particles left alone
		for (i=0; i<npart; ++i)
            if (particle[i].pointer!=i)
                flag++;
	}
	
    // Reset all cluster indexes
	for (i=0; i<npart; ++i)
        particle[i].i_cluster=-1;
	
	int icluster=0;
	for (i=0; i<npart; ++i)
	{
		if (particle[i].i_cluster==-1)
		{
		//Define spin properties
			dr=sqrt(Distance(particle[i], particle[particle[i].pointer])); //Distance returns dr^2
			phi=Pbca(Phi_spin(particle[i],particle[particle[i].pointer]));	
			xm=Pbcx(particle[i].x+Pbcx(particle[particle[i].pointer].x-particle[i].x, particle[particle[i].pointer].y-particle[i].y)/2., particle[i].y+Pbcy(particle[particle[i].pointer].y-particle[i].y)/2.);
			ym=Pbcy(particle[i].y+Pbcy(particle[particle[i].pointer].y-particle[i].y)/2.);
		
            //Assign spin values to both particles of the dimer
			particle[i].i_cluster=icluster;	
			particle[particle[i].pointer].i_cluster=icluster++;
			
			particle[i].xm=xm;	particle[particle[i].pointer].xm=xm;
			particle[i].ym=ym;	particle[particle[i].pointer].ym=ym;
			particle[i].dr=dr;	particle[particle[i].pointer].dr=dr;
			particle[i].phi=phi;	particle[particle[i].pointer].phi=phi;

			Particle appo = particle[i+1];
			particle[i+1] = particle[particle[i].pointer];
			particle[particle[i].pointer] = appo;
			particle[appo.pointer].pointer = particle[i].pointer;
			particle[i].pointer = i+1;
		}
	
	}
}

//==================
// Make_Spin
//==================

void Make_Spin(int& i)  // given a particle of index i, Make_Spin finds its closest neighbor
{

	int j_old, j_new, pj_new,j;
	double dr, dr_i, dr_j;

	j_old=particle[i].pointer;
	j_new=j_old;				// just to initialize
	
	if (i==particle[i].pointer) // if particle coupled with itself -> couple with the first one found free
	{
		for (j=0; j<npart; ++j)
			if (particle[j].pointer==j && i!=j) {j_new=j; break;}
	}
		
	dr_i = Distance(particle[i], particle[j_new]); // Trial value for the first comparison
	for (j=0; j<npart; ++j)			//check the distance with all the other particles and select the first one which is the closest one among those remained  free
	{
		if (j!=i)
		{
			dr =   Distance(particle[i], particle[j]);
			if (dr<dr_i)
			{
				if (particle[j].pointer==j) 
				{
					j_new=j;
					dr_i =dr;
				}
				else 
				{
					dr_j = Distance(particle[j], particle[particle[j].pointer]);
					if(dr<dr_j)
					{
						j_new=j;
						dr_i =dr;
					}
				}
			}
		}
	}
	if (j_new!=j_old) 			// if the algorithm has found a better companion for the particle i, update
	{
		pj_new=particle[j_new].pointer;
		if (pj_new!=j_new)		particle[pj_new].pointer=pj_new;
		particle[j_new].pointer=i;
			
		if (j_old!=i)			particle[j_old].pointer=j_old;
		particle[i].pointer=j_new;
	}

}

//======================
// Check_Configuration
//======================

void Check_Configuration()
{
	Cluster();

	int i_point;
    double dR, cosphi2, sinphi2;
	
	for (int i=0; i<npart; ++i) // loop over particles to find errors
	{
        cosphi2=cos(particle[i].phi)/2.;
        sinphi2=sin(particle[i].phi)/2.;
        
		if (particle[i].dr<r_hc) // dr is the distance between particles of a same cluster
		{
			i_point=particle[i].pointer;
			dR=r_hc*incr;
			
			particle[i].x=Pbcx(particle[i].xm+dR*cosphi2, particle[i].ym+dR*sinphi2); 		//Allontano tra di loro le particles in modo isotropo
			particle[i].y=Pbcy(particle[i].ym+dR*sinphi2);
				
			particle[i_point].x=Pbcx(particle[i].xm-dR*cosphi2, particle[i].ym-dR*sinphi2);
			particle[i_point].y=Pbcy(particle[i].ym-dR*sinphi2);
				
			particle[i].dr=dR;
			particle[i_point].dr=dR;
		}
	}
}

//===============
//Def_neighbors
//===============

void Def_neighbors() // Define neighbors lists
{
	double dR;
    int i,j;

	neighbors.resize(npart);

	for (i=0; i<npart-1; ++i)
	{
		for(j=i+1; j<npart; ++j)
		{
			dR=Distance(particle[i], particle[j]);
			if (dR<rverlet)
			{  // i and j are neighbors (i.e. within Verlet radius)
				neighbors[i].push_back(j);
				neighbors[j].push_back(i);
			}
		}
	}
}

//===============
//Measure
//===============

void Measure()
{
	int bin,i,j,k, nx2=nx/2, ny2=ny/2;
	long double v = 0.0, extra_v=0.0;
	double dr, dx, dy;

//reset the hystogram of g(r)
	for ( k=0; k<n_props; ++k) walker[k]=0.0;
	for ( k=0; k<nx2; ++k) bin_counter_x[k] = 0;
    for ( k=0; k<ny2; ++k) bin_counter_obl[k] = 0;
    
//cycle over pairs of particles
	for ( i=0; i<npart-1; ++i)
	{
		for ( j=i+1; j<npart; ++j)
		{
			dr= Distance(particle[i], particle[j]);
            //update of the histogram of g(r)
			bin = igofr + (int)(sqrt(dr)/bin_size);
			if(bin < igofr + nbins) walker[bin] += 2.0;
            //contribution to energy and virial
		    v += V_Pot(dr);
		    if (particle[i].i_cluster!=particle[j].i_cluster)
			extra_v+=V_Pot(dr); //exclude intra-cluster contribution
       	}
	}

	for ( i=0; i<npart-1; i+=2)
	{
		for( j=0; j<npart; j+=2)
		{
		if (i!=j){
            	if (int(particle[i].i_cluster/nx) == int(particle[j].i_cluster/nx)){ //if clusters are in the same horizontal row
            	    bin = igspin_x + Pbc_icl_x(particle[i].i_cluster,particle[j].i_cluster,nx)-1;
            	    walker[bin] += cos( 2.0 * (particle[i].phi - particle[j].phi) );
            	    bin_counter_x[bin-igspin_x]++;
            	}
            	else if ( (abs(particle[i].i_cluster - particle[j].i_cluster)) % nx == 0 ){ //if clusters are in the same diagonal line
                	bin = igspin_obl + Pbc_icl_obl(particle[i].i_cluster,particle[j].i_cluster,nx,ny)-1;
                    walker[bin] += cos( 2.0 * (particle[i].phi - particle[j].phi) );
        	        bin_counter_obl[bin-igspin_obl]++;
	            }
            }
		}
	}

	for ( i=0; i<nx2; ++i)
		if (bin_counter_x[i] != 0)
            		walker[i+igspin_x] /= (double)(bin_counter_x[i]);
    	for ( i=0; i<ny2; ++i)
        	if (bin_counter_obl[i] != 0)
            		walker[i+igspin_obl] /= (double)(bin_counter_obl[i]);

    walker[iv] = v;
	walker[iv2] = extra_v*extra_v; //exclude intra-cluster contribution from the specific heat, not necessary
	walker[iv3]= extra_v;
    
//Compute S(q) 
    double wvq,wvqx,wvqy;
    double sumc;
    double sums;
    double x_cart, y_cart, coord_obl;
    double qminox = qmino*cosang;
    double qminoy = qmino*sinang;

//Compute S(q) along x axis
   for ( i=0; i<nbinqx; ++i)
   {
	   	sumc = 0.0;
	   	sums = 0.0;
		wvq = (i+1)*qminx;
		for ( j=0; j<npart; j++)
		{
            x_cart = Pbc_cart_x (particle[j].x);
	   		sumc += cos(wvq * x_cart);
	   		sums += sin(wvq * x_cart);
	   	}
	   	walker[isofqx+i] = sumc * sumc + sums * sums;
   }
   
   //Compute S(q) along y axis
   for ( i=0; i<nbinqy; ++i)
   {
        sumc = 0.0;
        sums = 0.0;
        wvq = (i+1)*qminy;
        for ( j=0; j<npart; j++)
        {
            y_cart = Pbc_cart_y (particle[j].y);
            sumc += cos(wvq * y_cart);
            sums += sin(wvq * y_cart);
        }
        walker[isofqy+i] = sumc * sumc + sums * sums;
   }

//Compute S(q) along y' oblique axis
   for ( i=0; i<nbinqo; i++)
   {
        sumc = 0.0;
		sums = 0.0;
		wvqx = (i+1)*qminox;
		wvqy = (i+1)*qminoy;
		for ( j=0; j<npart; j++)
		{
            x_cart = Pbc_cart_x (particle[j].x);
		    y_cart = Pbc_cart_y (particle[j].y);
            sumc += cos(wvqx*x_cart+wvqy*y_cart);
            sums += sin(wvqx*x_cart+wvqy*y_cart);
		 }
		 walker[isofqo+i] = sumc * sumc + sums * sums;
    }

//Compute S(q) of dimers along x axis
   for ( i=0; i<nbinqx; ++i)
   {
	   	sumc = 0.0;
	   	sums = 0.0;
		wvq = (i+1)*qminx;
		for ( j=0; j<npart; j+=2)
		{
            x_cart = Pbc_cart_x (particle[j].xm);
	   		sumc += cos(wvq * x_cart);
	   		sums += sin(wvq * x_cart);
	   	}
	   	walker[isofq_clx+i] = sumc * sumc + sums * sums;
   }
   
   //Compute S(q) of dimers along y axis
   for ( i=0; i<nbinqy; ++i)
   {
        sumc = 0.0;
        sums = 0.0;
        wvq = (i+1)*qminy;
        for ( j=0; j<npart; j+=2)
        {
            y_cart = Pbc_cart_y (particle[j].ym);
            sumc += cos(wvq * y_cart);
            sums += sin(wvq * y_cart);
        }
        walker[isofq_cly+i] = sumc * sumc + sums * sums;
   }

    //Compute S(q) of dimers along y' oblique axis
    for ( i=0; i<nbinqo; i++)
    {
        sumc = 0.0;
		sums = 0.0;
		wvqx=(i+1)*qminox;
		wvqy=(i+1)*qminoy;
        
        for ( j=0; j<npart; j+=2)
        {
            x_cart = Pbc_cart_x (particle[j].xm);
		    y_cart = Pbc_cart_y (particle[j].ym);
            sumc += cos(wvqx*x_cart+wvqy*y_cart);
            sums += sin(wvqx*x_cart+wvqy*y_cart);
        }
		walker[isofq_clo+i] = sumc * sumc + sums * sums;
    }
}

//==================
// Reset
//==================

void Reset(int iblk) //Reset block averages
{
    int i;
	if(iblk == 1)
	{
		for(i=0; i<n_props; ++i)
		{
			glob_av[i] = 0.;
			glob_av2[i] = 0.;
		}
	}

	for(i=0; i<n_props; ++i)     blk_av[i] = 0.;
   
	blk_norm = 0.;
   
	for(i=0; i<n_moves; ++i)
	{
		accepted[i]=0;
		attempted[i]=0;
	}
}

//==================
// Reset_mpi
//==================

void Reset_mpi()  //reset to 0 accumulators for mpi blk averages
{
    int i;
    
	if(my_rank==rank_0)
	{
		for(i=0; i<n_props; ++i)
            mpi_blk_av[i] = 0.;
	}
	
	for(i=0; i<n_moves; ++i)
	{
		mpi_accepted[i]=0;
		mpi_attempted[i]=0;
	}
}

//==================
// Accumulate
//==================

void Accumulate(void) //Update block averages
{
	for(int i=0; i<n_props; ++i)
	{
		blk_av[i] += walker[i];
	}
	blk_norm = blk_norm + 1.0;
}

//==================
// Averages
//==================

void Averages(int iblk) //Print results for current block
{   
	long double err, r, gdir, gdis;
	long double stima;
	long double ene_2, ene;
    int kk, k;
 
	ofstream Gave, Gspin_x, Gspin_obl, Epot, Sofq_x, Sofq_y, Sofq_o, Sofq_clx, Sofq_cly, Sofq_clo, CS; //output streams
	cout << "Block number " << iblk << endl;
	cout << "Acceptance rate of delta00 " << double(mpi_accepted[0])/double(mpi_attempted[0]) << endl;
    cout << "Acceptance rate of delta20 " << double(mpi_accepted[1])/double(mpi_attempted[1]) << endl;
    cout << "Acceptance rate of delta50 " << double(mpi_accepted[2])/double(mpi_attempted[2]) << endl;
    cout << "Acceptance rate of rotation " << double(mpi_accepted[ir])/double(mpi_attempted[ir]) << endl;
    cout << "Acceptance rate of Break " << double(mpi_accepted[ib])/double(mpi_attempted[ib]) << endl;
    cout << "Acceptance rate of Cluster_Move " << double(mpi_accepted[icm])/double(mpi_attempted[icm]) << endl;
    
	Epot.open("output.epot.txt", ios::app);
	CS.open("output.specific_heat.txt", ios::app);
	if (iblk == nblk)
	{
		Gave.open("output.gave.txt");
		Gspin_x.open("output.gspin_x.txt");
        Gspin_obl.open("output.gspin_obl.txt");
		Sofq_x.open("output.sofq.x.txt");
        Sofq_y.open("output.sofq.y.txt");
		Sofq_o.open("output.sofq.o.txt");
		Sofq_clx.open("output.sofq.clx.txt");
        Sofq_cly.open("output.sofq.cly.txt");
		Sofq_clo.open("output.sofq.clo.txt");
	}
	
	//Potential energy
    stima = mpi_blk_av[iv]/mpi_blk_norm/(long double)npart + vtail; 
    glob_av[iv]  += stima;
    glob_av2[iv] += stima*stima;
    err=Error(glob_av[iv],glob_av2[iv],iblk);
    Epot << iblk << "   ";
    Epot << setprecision(prec) << glob_av[iv]/(long double)iblk << "   ";
    Epot << setprecision(prec)<< err << endl;
    
    //Specific_Heat
   	ene_2= mpi_blk_av[iv2]/mpi_blk_norm;
   	ene  = mpi_blk_av[iv3]/mpi_blk_norm;
   	stima = (ene_2-ene*ene)/(long double)(temp*temp*npart); 
    glob_av[iv2] += stima;
    glob_av2[iv2] += stima*stima;
    err=Error(glob_av[iv2],glob_av2[iv2],iblk);
    CS << iblk << "   ";
    CS <<setprecision(prec) << glob_av[iv2]/(long double)iblk << "   ";
    CS <<setprecision(prec) << err << endl;

    //g(r)
    for (k=igofr; k<igofr+nbins; ++k)
    {
		kk = k - igofr;
		r = kk * bin_size; //bin goes from r to r+dr
		gdir = mpi_blk_av[k]/mpi_blk_norm;
		gdir *= 1.0/(pi * ((r + bin_size)*(r + bin_size) - r*r) * rho * (long double)npart);
		glob_av[k] += gdir;
		glob_av2[k] += gdir*gdir;
		if(iblk == nblk)
		{
			err=Error(glob_av[k],glob_av2[k],iblk);
			Gave << setprecision(prec) << ((double)kk + 0.5)*bin_size 	<< "   ";
			Gave << setprecision(prec) << glob_av[k]/(double)iblk 		<< "   ";
			Gave << err << "   " << endl;
		}
    }

    //gspin_x - spin spin correlation functions
	
	Gspin_x  << 0 << '\t' << 1 << '\t' << 0 << endl; //explicitly add self-correlation, bin 0
    for (k=igspin_x; k<igspin_obl; ++k)
    {
		kk = k - igspin_x;
		gdis = mpi_blk_av[k]/mpi_blk_norm;
		glob_av[k] += gdis;
		glob_av2[k] += gdis*gdis;
		if(iblk == nblk)
		{
			err=Error(glob_av[k],glob_av2[k],iblk);
			Gspin_x << kk+1 << "   ";
			
			Gspin_x << setprecision(prec) << glob_av[k]/(double)iblk 		<< "   ";
			Gspin_x << err << "   " << endl;
		}
    }

    //gspin_obl
        
    Gspin_obl  << 0 << '\t' << 1 << '\t' << 0 << endl; //explicitly add self-correlation, bin 0
    for (k=igspin_obl; k<isofqx; ++k) 
    {
        kk = k - igspin_obl;
        gdis = mpi_blk_av[k]/mpi_blk_norm;
        glob_av[k] += gdis;
        glob_av2[k] += gdis*gdis;
        if(iblk == nblk)
        {
            err=Error(glob_av[k],glob_av2[k],iblk);
            Gspin_obl << kk+1 << "   ";
            Gspin_obl << setprecision(prec) << glob_av[k]/(double)iblk 		<< "   ";
            Gspin_obl << err << "   " << endl;
        }
    }


    //S(q) along x axis
	double wvq;
	double sdik;
    for (k=isofqx; k<isofqx+nbinqx; k++)
    {
       	wvq = (k-isofqx+1)*qminx;
       	sdik = mpi_blk_av[k]/(mpi_blk_norm*(double)(npart));	//block average for S(q) along x axis
       	glob_av[k]  += sdik;
       	glob_av2[k] += sdik * sdik;
       	if(iblk == nblk)
		{
			err = Error(glob_av[k],glob_av2[k],iblk);
            Sofq_x << setprecision(prec) <<  wvq << "   " << glob_av[k]/(double)iblk << "  " << err << "     " << endl;
		}
   	}
    
    //S(q) along y axis
    for (k=isofqy; k<isofqy+nbinqy; k++)
    {
        wvq = (k-isofqy+1)*qminy;
        sdik = mpi_blk_av[k]/(mpi_blk_norm*(double)(npart));    //block average for S(q) along y axis
        glob_av[k]  += sdik;
        glob_av2[k] += sdik * sdik;
        if(iblk == nblk)
        {
            err = Error(glob_av[k],glob_av2[k],iblk);
            Sofq_y << setprecision(prec) <<  wvq << "   " << glob_av[k]/(double)iblk << "  " << err << "     " << endl;
        }
    }

    //S(q) along y' axis - oblique axis
	for (k=isofqo; k<isofqo+nbinqo; k++)
	{
		wvq = (k-isofqo+1)*qmino;
		sdik = mpi_blk_av[k]/(mpi_blk_norm*(double)(npart));	//block average for S(q) along y' axis
		glob_av[k]  += sdik;
		glob_av2[k] += sdik * sdik;
		if(iblk == nblk)
		{
			err = Error(glob_av[k],glob_av2[k],iblk);
         	Sofq_o << setprecision(prec) <<  wvq << "   " << glob_av[k]/(double)iblk << "  " << err << endl;
    	}
	} 

    //S(q) along x axis - cluster
    for (k=isofq_clx; k<isofq_clx+nbinqx; k++)
    {
       	wvq = (k-isofq_clx+1)*qminx;
       	sdik = 2*mpi_blk_av[k]/(mpi_blk_norm*(double)(npart));	//block average for S(q)_cl along x axis
       	glob_av[k]  += sdik;
       	glob_av2[k] += sdik * sdik;
       	if(iblk == nblk)
		{
			err = Error(glob_av[k],glob_av2[k],iblk);
            Sofq_clx << setprecision(prec) <<  wvq << "   " << glob_av[k]/(double)iblk << "  " << err << "     " << endl;
		}
   	}
    
    //S(q) along y axis - cluster
    for (k=isofq_cly; k<isofq_cly+nbinqy; k++)
    {
        wvq = (k-isofq_cly+1)*qminy;
        sdik = 2*mpi_blk_av[k]/(mpi_blk_norm*(double)(npart));    //block average for S(q)_cl along y axis
        glob_av[k]  += sdik;
        glob_av2[k] += sdik * sdik;
        if(iblk == nblk)
        {
            err = Error(glob_av[k],glob_av2[k],iblk);
            Sofq_cly << setprecision(prec) <<  wvq << "   " << glob_av[k]/(double)iblk << "  " << err << "     " << endl;
        }
    }

//S(q) along y' axis - oblique axis - cluster
	for (k=isofq_clo; k<isofq_clo+nbinqo; k++)
	{
		wvq = (k-isofq_clo+1)*qmino;
		sdik = 2*mpi_blk_av[k]/(mpi_blk_norm*(double)(npart));	//block average for S(q)_cl along y' axis
		glob_av[k]  += sdik;
		glob_av2[k] += sdik * sdik;
		if(iblk == nblk)
		{
            err = Error(glob_av[k],glob_av2[k],iblk);
            Sofq_clo << setprecision(prec) <<  wvq << "   " << glob_av[k]/(double)iblk << "  " << err << endl;
        }
	}
    cout << "----------------------------" << endl << endl;

    Epot.close();
    CS.close();
	if (iblk == nblk)
	{
    	Gave.close();
		Gspin_x.close();
        Gspin_obl.close();
    	Sofq_x.close();
        Sofq_y.close();
    	Sofq_o.close();
        Sofq_clx.close();
        Sofq_cly.close();
        Sofq_clo.close();
    }
}

//==================
// ConfFinal
//==================

void ConfFinal(void) //print final configuration in units normalized wrt box side
{
	ofstream file_output1, file_output2, WriteSeed, WriteRestart;
	
    //Save last seed
	WriteSeed.open("output.seed");
	SaveRandom(seed);
	WriteSeed << seed[0] << " " << seed[1] << " " << seed[2] << " " << seed[3] << endl << endl;
	WriteSeed.close();

    //Save restart parameters
	WriteRestart.open("restart.dat");
	WriteRestart << nblk << endl;
	WriteRestart << setprecision(15) << u1x << endl;
	WriteRestart << setprecision(15) << u2x << endl;
	WriteRestart << setprecision(15) << u2y << endl;
	WriteRestart << nx << endl;
	WriteRestart << ny << endl;
	WriteRestart << ncluster << endl;
	WriteRestart << rho << endl;
    WriteRestart << angle << endl;

	WriteRestart << setprecision(15) << glob_av[iv] << "    " << glob_av2[iv] << endl;
	WriteRestart << setprecision(15) << glob_av[iv2] << "    " << glob_av2[iv2] << endl;
    
	WriteRestart.close();

	cout << "Print final configuration " << endl << endl;

	file_output1.open("output.config.final.txt");				// Units divided by box side
	file_output2.open("output.aconfig.txt");					// Print spin xm, ym, phi, dr

	for (int i=0; i<npart; ++i)
		file_output1 << particle[i].x/boxx << "   " <<  particle[i].y/boxy << "     " << particle[i].i_cluster << endl;

    //Spin in normalized box
    //It is wise to re-evaluate clusters composition
	Cluster();
	
	int k=-1,i,j;
    double constant = 180./pi;
	for (i=0; i<nspin; ++i)
	{
		for (j=0; j<npart; ++j)
			if (particle[j].i_cluster==i)
                k=j;
                
		file_output2 << setprecision(15) << particle[k].xm/boxx << "   " <<  particle[k].ym/boxy <<"   ";
		file_output2 << setprecision(15) << constant*particle[k].phi<<"   "<<particle[k].dr<< endl;
	}
	
	file_output1.close();
	file_output2.close();
}

//==================
// ConvertStr
//==================

string ConvertStr (int& n)  //convert integers to strings
{
    ostringstream a;
    a << n;
    return a.str();
}

//==================
// ConfPartial
//==================

void ConfPartial(int& n, bool check) //print intermediate system configurations scaled by box sides
{
	ofstream file_output;
	string name_file;
	
	if (check)	 	name_file = "output.econfig."+ConvertStr(my_rank)+"."+ConvertStr(n)+".txt";
	else			name_file = "output.config."+ConvertStr(my_rank)+"."+ConvertStr(n)+".txt";
	
	file_output.open(name_file.c_str());
	for (int i=0; i<npart; ++i)
 		file_output<<particle[i].x/boxx<<"	"<<particle[i].y/boxy<<"	"<<particle[i].i_cluster<<"	"<<endl;	
	
	file_output.close();
}

//====================
// Print_Parameters
//====================

void Print_Parameters()
{
    //Evaluate potential energy and virial of the initial configuration
	Measure();

    //Print initial values for the potential energy and virial
	cout << "Initial potential energy (with tail corrections) = " << setprecision(prec) << walker[iv]/(long double)npart + vtail << endl;
}

//==================
// Pbcx
//==================

double Pbcx(double x, double y)  //Algorithm for periodic boundary conditions with rhomboic box
{
    return x - boxx * rint((x-y/tga)/boxx) - boxy/tga * rint(y/boxy);
}

//==================
// Pbcy
//==================

double Pbcy(double r)  //Algorithm for periodic boundary conditions with side L=box
{
    return r - boxy * rint(r/boxy);
}

//==================
// Pbc_cart_x
//==================

double Pbc_cart_x(double r)  //Algorithm for periodic boundary conditions - carthesian frame of reference
{
    return r - my_boxx * rint(r/my_boxx);
}

//==================
// Pbc_cart_y
//==================

double Pbc_cart_y(double r)  //Algorithm for periodic boundary conditions - carthesian frame of reference
{
    return r - boxy * rint(r/boxy);
}


//==================
// Pbca
//==================

double Pbca(double a)  //Algorithm for periodic boundary conditions with angular pi
{
	double angle = a -  pi* rint(a/pi);
	if (angle > 0)
        return angle;
	else
        return angle + pi;
}

//==================
// Pbc_icl_x
//==================

int Pbc_icl_x(int a, int b, int nx){
    int dist=abs(b-a);
    if (dist>nx/2)
        dist=nx-dist;
  return dist;
}

//==================
// Pbc_icl_obl
//==================

int Pbc_icl_obl(int a, int b, int nx, int ny){
    int dist=abs(b-a)/nx; //a row is nx-long
    if (dist>ny/2)
        dist=ny-dist;
    return dist;
}

//==================
// Error
//==================

long double Error(long double sum,long  double sum2, int iblk) // data blocking error
{
    if (iblk==1)
        return 0.;
    else
        return sqrt((sum2/(long double)iblk - (sum/(long double)iblk)*(sum/(long double)iblk))/(long double)(iblk-1));
}

//==================
// V_Pot
//==================

//--------------------Generic potential function----------
long double V_Pot(double r)
{
	long double energy =0.;
	if (r<rcut2)
	{
		energy = exp(-1.*r*r);
		if (r<r_hc2)
		{
			energy +=khc;
		}
	}
	return energy;
}

//==================
// V_Tail
//==================

long double V_Tail()
{
	return pi*rho*sqrt(pi)*erfc(rcut*rcut)/4.;
}

//===================
// Distance
//===================

double Distance(Particle& par1, Particle& par2) //compute square distance between par1 and par2
{
	double dx, dy, dr;
	dx = Pbcx(par1.x - par2.x, par1.y - par2.y);
	dy = Pbcy(par1.y - par2.y);

	dr = dx*dx + dy*dy;
	return dr;
}

//===================
// Phi_spin
//===================


double Phi_spin(Particle& par1, Particle& par2) //return phi degree of freedom for spins
{
	double mi, dx, dy;
	dx= Pbcx(par1.x-par2.x, par1.y-par2.y);
	dy= Pbcy(par1.y-par2.y);
	
	if (dx==0)
	{
		if(p_check) {mi=pi/2.0; p_check=false;}
		else		{mi=pi/(-2.0); p_check=true;}
	}
	else mi=atan(dy/dx);

	return Pbca(mi);
}

// Random numbers -------------------------------------------------

//=======================
// Gauss
//=======================

double Gauss(int n1, int n2, int n3, int n4)
{
	double v1,v2,rsq,fac,g;

	if(igauss == 0)
	{
        do
        {
            v1 = 2.0*Rannyu() - 1.0;
            v2 = 2.0*Rannyu() - 1.0;
            rsq = v1*v1 + v2*v2;
        }while(rsq >= 1.0 || rsq == 0);

        fac = sqrt(-2.0*log(rsq)/rsq);
        g1 = v1*fac;
        g2 = v2*fac;
        g = g2;
        igauss = 1;
  	}

	else
  	{
        g = g1;
        igauss = 0;
    }

  	return g;
}

//===============
// Rannyu
//===============

double Rannyu() // generate random numbers between 0 and 1
{
	const long double twom12=0.000244140625;
  	int i1,i2,i3,i4;
  	double r;

  	i1 = l1*m4 + l2*m3 + l3*m2 + l4*m1 + n1;
  	i2 = l2*m4 + l3*m3 + l4*m2 + n2;
  	i3 = l3*m4 + l4*m3 + n3;
  	i4 = l4*m4 + n4;
  	l4 = i4%4096;
  	i3 = i3 + i4/4096;
  	l3 = i3%4096;
  	i2 = i2 + i3/4096;
  	l2 = i2%4096;
  	l1 = (i1 + i2/4096)%4096;
  	r=twom12*(l1+twom12*(l2+twom12*(l3+twom12*(l4))));

  	return r;
}

//=======================
// SetRandom
//=======================

void SetRandom(int * s, int p1, int p2)
{
	m1 = 502;
  	m2 = 1521;
  	m3 = 4071;
  	m4 = 2107;
  	l1 = s[0]%4096;
  	l2 = s[1]%4096;
  	l3 = s[2]%4096;
  	l4 = s[3]%4096;
  	l4 = 2*(int(l4/2))+1;
 	n1 = 0;
 	n2 = 0;
 	n3 = p1;
 	n4 = p2;

}

//=======================
// SaveRandom
//=======================

void SaveRandom(int * s)
{
  	s[0] = l1;
  	s[1] = l2;
	s[2] = l3;
  	s[3] = l4;
}
/****************************************************************
*****************************************************************
    _/      _/_/_/  _/_/_/  Laboratorio di Calcolo Parallelo e
   _/      _/      _/  _/  di Simulazioni di Materia Condensata
  _/      _/      _/_/_/  c/o Sezione Struttura della Materia
 _/      _/      _/      Dipartimento di Fisica
_/_/_/  _/_/_/  _/      Universita' degli Studi di Milano
*****************************************************************
*****************************************************************/
