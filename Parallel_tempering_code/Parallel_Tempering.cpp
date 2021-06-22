/****************************************************************
*****************************************************************
    _/      _/_/_/  _/_/_/  Laboratorio di Calcolo Parallelo e
   _/      _/      _/  _/  di Simulazioni di Materia Condensata
  _/      _/      _/_/_/  c/o Sezione Struttura della Materia
 _/      _/      _/      Dipartimento di Fisica
_/_/_/  _/_/_/  _/      Universita' degli Studi di Milano
*****************************************************************
*****************************************************************/

/* 22/06/2021

Code version: 7.0 (internal code development of the group)

Contributions by: Fabio Civillini, Matteo Martinelli, Francesco Mambretti, Davide E. Galli

MPI code: Parallel tempering simulation for the equilibrium phase of 2D soft matter in the n=2 cluster phase. Each rank works on a different trajectory found at a different temperature, and they periodically try to exchange configurations

Prints a file, as output, named output.MC.lattice.txt, to be used as input.lattice.dat in a subsequent  Monte Carlo simulation
        
*/
// GITHUB VERSION




#include"Parallel_Tempering.h"
#include"Random.h"

int main(int argc, char** argv)
{ 
    MPI_Init(&argc, &argv);
	
//measure initial time for performance
	double start, end, startb, endb, timexblk;
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

// Build a vector in the control rank to collect acceptance ratios
	if (my_rank==rank_0)
        Create_Array();

// Initialization
	Input();
	if (my_rank==rank_0) 
	{
		cout << "E_rank_0 = "<< setprecision(prec) << ene << endl;
		cout << "E/N_rank_0 = " << setprecision(prec) << ene/(ndimer * 2.0) << endl;
	}
	
	Print_Energy(0);
	Print_Box(0);
	startb = MPI_Wtime();
	for(int iblk=1; iblk <= nblk; ++iblk)
 	{
		Reset(iblk);   										//Reset block averages
		for(int istep=1; istep <= nstep; ++istep) 
		{
		// Evolve the whole system within each rank
			Move();
			Rotate();
			Break();
			Aspect_Ratio();
			Angle();
		}
		
		Print_Energy(iblk);
		Print_Box(iblk);

		MPI_Barrier(MPI_COMM_WORLD);	
		
        // Gather final configuration energies into rank_0
		MPI_Gather(&ene, 1, MPI_LONG_DOUBLE, energy, 1, MPI_LONG_DOUBLE, rank_0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	
	    // Try to exchange the configurations of the various ranks
		Exchange();
		MPI_Barrier(MPI_COMM_WORLD);
		
        // Infer (approximately) time to complete the operations
		endb = MPI_Wtime();	  
		if (my_rank==rank_0)
		{
			timexblk=(endb-startb)/(double)iblk;
			cout <<"blocco "<< iblk <<" su " << nblk<<"   ";
			int h, min, s;
			double tot = timexblk*(double)(nblk-iblk);
			h = tot/3600;
			min = (tot - h*3600.)/60;
			s = (tot - h*3600. - min * 60.);
			cout <<"t stimato = "<<setprecision(3)<< h <<" h " << min << " min " << s << " s 	(" << tot << " s)"<< endl;
		} 	
	}
	
    //Gather acceptance rates into rank_0
	MPI_Gather(&acceptedM, 1, MPI_DOUBLE, accepted_M, 1, MPI_DOUBLE, rank_0, MPI_COMM_WORLD);
	MPI_Gather(&acceptedR, 1, MPI_DOUBLE, accepted_R, 1, MPI_DOUBLE, rank_0, MPI_COMM_WORLD);
	MPI_Gather(&acceptedB, 1, MPI_DOUBLE, accepted_B, 1, MPI_DOUBLE, rank_0, MPI_COMM_WORLD);
	MPI_Gather(&acceptedAR, 1, MPI_DOUBLE, accepted_AR, 1, MPI_DOUBLE, rank_0, MPI_COMM_WORLD);
	MPI_Gather(&acceptedA, 1, MPI_DOUBLE, accepted_A, 1, MPI_DOUBLE, rank_0, MPI_COMM_WORLD);

	if (my_rank==rank_0)
	{
	//Print results
		Switch_Rate();
		Temperature_Rate();
	}
	
	Print_Config(my_rank, false);
	
	MPI_Barrier(MPI_COMM_WORLD);
	Delete_Array();
		
    //measure end time and print results
	MPI_Barrier(MPI_COMM_WORLD);
	end=MPI_Wtime();
	if (my_rank==rank_0)
    {
        cout <<size<<" time:  "<<end-start<<endl;
    }
	
	MPI_Finalize();
	return 0;
}

//==================
// Input
//==================

void Input()
{
// Give to each rank a different set of parameters (seed, temperature, rotation angle)
	ifstream Primes, Temp, Ang, Del, Del_b, Del_AR, Del_a;
	Primes.open("Primes");
	Temp.open("Temperature.dat");
	Ang.open("Alpha.dat");
	Del.open("Delta.dat");
	Del_b.open("Delta_b.dat");
	Del_AR.open("Delta_AR.dat");
	Del_a.open("Delta_a.dat");
  	
  	int p1,p2,count=0;
  	while(count<= my_rank) 
  	{
		Temp >> temp ;
		Ang >> alpha;
		Del >> delta;//move
		Del_b >> delta_b;//break
		Del_AR >> delta_AR;
		Del_a >> delta_a;
		Primes >> p1 >> p2 ;
		
		beta=1./temp;
		count++;  
	}
 	
 	Primes.close();
 	Ang.close();
 	Temp.close();
 	Del.close();
 	Del_b.close();
	Del_AR.close();
	Del_a.close();

// Master rank gathers info on the simulated temperatures
 	MPI_Gather(&beta, 1, MPI_LONG_DOUBLE, beta_vec, 1, MPI_LONG_DOUBLE, rank_0, MPI_COMM_WORLD);

 	if (my_rank==rank_0) //only rank master
	{
		ifstream ReadSeed;
	
	//Read seed for random numbers
		ReadSeed.open("seed");
		ReadSeed >> seed[0] >> seed[1] >> seed[2] >> seed[3];
		ReadSeed.close();
	}
 	
// Set RNG for each rank
 	MPI_Bcast(seed,4,MPI_INT,rank_0,MPI_COMM_WORLD);
	SetRandom(seed,p1,p2);
	
	if (my_rank==rank_0) //only rank master
	{
	//Read input informations
		ifstream ReadInput, ReadLattice;
		ReadInput.open("input.dat");
	
		ReadInput >> m_lattice;
		ReadInput >> s_lattice;
		ReadInput >> khc; 	
		ReadInput >> rhc;				
        rhc2=rhc*rhc;
        
		ReadInput >> rcut;
        rcut=rcut*rcut;
        
		ReadInput >> rverlet;
        rverlet=rverlet*rverlet;
        
		ReadInput >> nblk;
		ReadInput >> nstep;
		ReadInput >> nval; //degen.
		
		ReadInput.close();

		ReadLattice.open("input.lattice.dat");

		ReadLattice >> u1x;
		ReadLattice >> u2x;
		ReadLattice >> u2y;
		ReadLattice >> ndimer;
		ReadLattice >> nx; // number of elem. cells along x
		ReadLattice >> ny; // and y
		ReadLattice >> rho;

		ReadLattice.close();

		int npart = 2*ndimer;
        //Normalize rectangular cells
		u2x /= u1x;
		u2y /= u1x;
		u1x = 1.0;
		tga = u2y/u2x;
		cout << "Bases vectors u1x, u2x and u2y " << u1x << '\t' << u2x << '\t' << u2y << endl;

		int ncells = nx*ny;
		cout << "Number of cells = " << nx << " x " << ny << endl;
		double scell = u1x * u2y;  //area of normalized elementary cell
		vol = (double)npart/rho;
		double acell = sqrt(vol/((double)(ncells)*scell)); // scale factor to physical units
		u1x *= acell;
		u2y *= acell;
		u2x *= acell;
		boxx = (double)nx * u1x;
		boxy = (double)ny * u2y;
		cout<<" Simulation box sides: "<<setprecision(4)<<boxx<<" x "<<boxy<<endl;
		
//		if (m_lattice !=0) Num_dimer();

		cout << "Number of dimers = " << ndimer << endl;
		cout << "Numerical density = " << rho << endl;
	}
	
// Broadcast all input information to each rank
	MPI_Bcast(&m_lattice,	1,MPI_INT,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&s_lattice,	1,MPI_LOGICAL,rank_0,MPI_COMM_WORLD);

	MPI_Bcast(&ndimer,		1,MPI_INT,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&nx,			1,MPI_INT,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&ny,			1,MPI_INT,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&vol,			1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&u1x,			1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&u2x,			1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&u2y,			1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&tga,			1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&boxx,		1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&boxy,		1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);

	MPI_Bcast(&khc,			1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&rhc,			1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&rhc2,		1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&rho,			1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&rcut,		1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&rverlet,		1,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	
	MPI_Bcast(&nblk,		1,MPI_INT,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&nstep,		1,MPI_INT,rank_0,MPI_COMM_WORLD);
	MPI_Bcast(&nval,		1,MPI_INT,rank_0,MPI_COMM_WORLD);
	
// Allocate dynamic arrays containing dimer info
	xm = new double [ndimer]; // barycenter
	ym = new double [ndimer];
	phi = new double [ndimer];
	dr = new double [ndimer];
	park_xm = new double [ndimer]; // temporary variable for the exchange move
	park_ym = new double [ndimer];
	park_phi = new double [ndimer];
	park_dr = new double [ndimer];

	park_boxx = 0;
	park_boxy = 0;
	park_tga = 0;
	
//Generate lattice
	if (s_lattice)
	{
		if (my_rank==rank_0) 
		{
			Make_Lattice();
			cout << "All the ranks start with the same initial configuration" << endl;
		}	
		MPI_Bcast(xm,ndimer,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
		MPI_Bcast(ym,ndimer,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
		MPI_Bcast(phi,ndimer,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
		MPI_Bcast(dr,ndimer,MPI_DOUBLE,rank_0,MPI_COMM_WORLD);
	}
	else 
	{
		Make_Lattice();
		if (my_rank==rank_0)
            cout << "Each rank starts from a different configuraton" << endl;
	}
	if (my_rank==rank_0)
        cout<<"The barycenter of the dimer which remains fixed has coordinates: ("<<xm[0]<<";"<<ym[0]<<")"<<endl;

	MPI_Barrier(MPI_COMM_WORLD);
	Def_Neighbors(); //Verlet - evaluate neighbors lists

// Compute total potential energy
	ene=E_Pot_Tot();
}

//==================
// Reset
//==================

void Reset(int iblk) //Reset block averages
{
	ene_av=0.0;
	blok_norm=0.0;
	
	if (iblk==1)
	{
		for (int i=0; i<size-1; ++i)
		{
			if (my_rank==rank_0)
            {
				accepted_S[i]=0.;
				attempted_S[i]=1.;
            }
			attemptedM	=0;
			acceptedM	=0;
			attemptedR	=0;
			acceptedR	=0;
			attemptedB	=0;
			acceptedB	=0;
			attemptedAR	=0;
			acceptedAR	=0;
			attemptedA  =0;
			acceptedA   =0;		
			
		}
		global_av=0.0;
		global_av2=0.0;
	}
}

//==================
// Accumulate
//==================

void Accumulate()
{
	ene_av+=ene;
	blok_norm+=1.0;
}

//==================
// Averages
//==================

void Averages(int iblk)
{
	double error;
	global_av+=ene_av/blok_norm;
	global_av2=global_av2+global_av*global_av;
	error=Error(global_av, global_av2, iblk);
	
	ofstream file_output;
	string nome_file;
	nome_file = "output.epot."+ConvertStr(my_rank)+".txt";
	file_output.open(nome_file.c_str(), ios::app);
	file_output <<iblk<<"   "<<setprecision(prec)<<ene_av/blok_norm<<"   "<<global_av/(double)iblk<<"   "<<error<<endl;
	file_output.close();
}

//==================
// Error
//==================

double Error(double& sum, double& sum2, int& iblk)
{
    return sqrt((sum2/(double)iblk - (sum/(double)iblk)*(sum/(double)iblk))/(double)iblk);
}

//==================
// Move
//==================

void Move()
{
	int o;															// index of the moved dimer
	double xx, yy;													// barycenter coordinates
	double p;														// acceptance probability
	double energy_old, energy_new, d_ene;							// energies, and delta energy
	
	for(int i=0; i<ndimer; ++i)	// try ndimer moves
	{
	//Select randomly a dimer
	// Note that the 0-indexed dimer is fixed and cannot be moved
    
		o = (Rannyu()*(ndimer-1))+1;
   		
	//Old
		energy_old = Boltzmann_M(xm[o], ym[o], o);

	//New
		xx = delta*(Rannyu()-0.5)+xm[o];				// displacement in the range [-delta/2; delta/2]
		yy = delta*(Rannyu()-0.5)+ym[o];				// displacement in the range [-delta/2; delta/2]

		xx = Pbcx(xx, yy);
		yy = Pbcy(yy);
		energy_new = Boltzmann_M(xx,yy, o);

	//Metropolis test
		d_ene=energy_new-energy_old;
		p = exp(-beta*d_ene);
		if(p >= Rannyu())  
		{
        //Update
			xm[o]=xx;
			ym[o]=yy;
			ene+=d_ene;
			acceptedM += 1.0;
		}
		attemptedM += 1.0;
	}
}

//==================
// Rotate
//==================

void Rotate()
{
    int o;                                                            // index of the moved dimer
    double theta;                                                    // angle
    double p;                                                        // acceptance probability
    double energy_old, energy_new, d_ene;                            // energies, and delta energy
	
	for(int i=0; i<ndimer; ++i)    // try ndimer moves
	{
        //Select randomly a dimer
		o = (Rannyu()*ndimer);
   		
	//Old
		energy_old = Boltzmann_R(phi[o], o);

	//New
		theta=Pbca(alpha*(Rannyu()-0.5)+phi[o]);					//generate angle in the range [-alpha/2; alpha/2]
		energy_new = Boltzmann_R(theta, o);

	//Metropolis test
		d_ene=energy_new-energy_old;
		p = exp(-beta*d_ene);
		if(p >= Rannyu())  
		{
	//Update
			phi[o]=theta;
			ene+=d_ene;
			acceptedR += 1.0;
		}
		attemptedR += 1.0;
	}
}

//==================
// Break
//==================

void Break()
{
    int o;                                                            // index of the moved dimer
    double dR;                                                    // inner distance between the two particles of a dimer
    double p;                                                        // acceptance probability
    double energy_old, energy_new, d_ene;                            // energies, and delta energy
	
	for(int i=0; i<ndimer; ++i)	// try ndimer moves
	{
	//Select randomly a dimer
		o = (Rannyu()*(ndimer));
   		
	//Old
		energy_old = Boltzmann_B(dr[o], o);

	//New
		dR=delta_b*(Rannyu()-0.5)+dr[o];							//generate dilation in the range [-delta/2; delta/2]
		energy_new = Boltzmann_B(dR, o);

	//Metropolis test
		d_ene=energy_new-energy_old;
		p = exp(-beta*d_ene);
		if(p >= Rannyu())  
		{
	//Update
			dr[o]=dR;
			ene+=d_ene;
			acceptedB += 1.0;
		}
		attemptedB += 1.0;
	}
}


//==================
// Aspect Ratio
//==================

void Aspect_Ratio()
{
	
	double old_boxx, old_boxy, new_boxx, new_boxy;                  // new and old box sides
	double* old_xm;
	double* old_ym;
	double dilat;
	double p;														//acceptance probability
	double energy_old, energy_new, d_ene;

    // Save original values
	old_boxx = boxx;
	old_boxy = boxy;
	energy_old = ene;

	old_xm = new double [ndimer];
	old_ym = new double [ndimer];

    //Change box aspect ratio
	new_boxx = boxx + delta_AR * (Rannyu() - 0.5);		//change the old boxx values
	new_boxy = vol/new_boxx;		//keep total density (and volume) constant
	dilat = new_boxx/old_boxx;

    // update sides
	boxx = new_boxx;
	boxy = new_boxy;

	for (int i=0; i<ndimer; ++i)	//set dimers in the new lattice
	{						
	//Save old values
		old_xm[i] = xm[i];
		old_ym[i] = ym[i];
	// Set the new ones
		xm[i] = Pbcx (xm[i] * dilat, ym[i]/dilat);
		ym[i] = Pbcy (ym[i]/dilat);
	}

//Metropolis test
	energy_new = E_Pot_Tot();
	d_ene=energy_new-energy_old;
	p = exp(-beta*d_ene);
	if(p >= Rannyu())  			// accepted
	{
	//Update
		ene = energy_new;
		acceptedAR += 1.0;
	}
	else 							// bring back to the old lattice
	{
		ene = energy_old;
		boxx = old_boxx;
		boxy = old_boxy;
		for(int i=0; i<ndimer; ++i)	// and to the old dimer positions
			{
				xm[i] = old_xm[i];
				ym[i] = old_ym[i];
			}	
	}
	attemptedAR += 1.0;
	delete [] old_xm;
	delete [] old_ym;
}


//==================
// Angle
//==================

void Angle ()
{
	
	double old_tga, new_tga, inv_old_tga, inv_new_tga;
	double* old_xm;
	double disp;													//displacement induced by the new angle
	double p;														// acceptance probability
	double energy_old, energy_new, d_ene;							// old and new energies

// Original values
	old_tga = tga;
	energy_old = ene;

	old_xm = new double [ndimer];

//Change the angle comprised between the two box bases vectors
	new_tga = tga + delta_a * (Rannyu() - 0.5);
	tga = new_tga;
    inv_new_tga = 1.0/new_tga;
    inv_old_tga = 1.0/old_tga;

// This move only affects x coordinate
	for(int i=0; i<ndimer; ++i)
	{
		old_xm[i] = xm[i];

		disp = ym[i] * (inv_new_tga - inv_old_tga);
		xm[i] = Pbcx (xm[i] + disp, ym[i]);
	}

//Metropolis test
	energy_new = E_Pot_Tot();
	d_ene=energy_new-energy_old;
	p = exp(-beta*d_ene);
	if(p >= Rannyu())  				// if the move is accepted, update lattice
	{
	//Update
		ene = energy_new;
		acceptedA += 1.0;
	}
	else 							//otherwise bring back to the old configuration
	{
		for (int i=0; i<ndimer; ++i)
            xm[i] = old_xm[i];
		ene = energy_old;
		tga = old_tga;
	}
	attemptedA += 1.0;
}

//==================
// Exchange
//==================

void Exchange()
{
	double p;
	int j;							//index of the rank with which the exchange is performed
	double park_ene;
	
//Start from the highest temperature and then lower it
	for (int i=size-1; i>0; --i)
	{
		j=i-1;
	
	// Rank 0 knows all the data, so evaluate the exchange probability
		if (my_rank==rank_0)
		{
			p_switch= exp((beta_vec[j]-beta_vec[i])*(energy[j]-energy[i]));
			r_switch= Rannyu();
		}
	
	// Tell all the ranks about this opportunity to perform the exchange
		MPI_Bcast(&p_switch,1,MPI_LONG_DOUBLE,rank_0,MPI_COMM_WORLD);
		MPI_Bcast(&r_switch,1,MPI_LONG_DOUBLE,rank_0,MPI_COMM_WORLD);
		
		if(p_switch >= r_switch)
		{
		// Switch j and i configurations
			Switch(j,i); 
		
		// Update
			if (my_rank==rank_0)
			{
				park_ene=energy[j];
				energy[j]=energy[i];
				energy[i]=park_ene;
				accepted_S[j]+=1.0;
			}
		}
		
		if (my_rank==rank_0)
            attempted_S[j]+=1.0;
		
		MPI_Barrier(MPI_COMM_WORLD); //synchronize
	}
}

//==================
// Switch
//==================

void Switch(int& left, int& right)
{
	
//Copy vectors with observables into temporary vectors
	if (my_rank==left)
	{
		if (left==size-1) cout <<"error on last renk"<<endl;
		
		MPI_Send(xm, ndimer, MPI_DOUBLE, my_rank+1, 2, MPI_COMM_WORLD);
		MPI_Recv(park_xm, ndimer, MPI_DOUBLE, my_rank+1, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		
		MPI_Send(ym, ndimer, MPI_DOUBLE, my_rank+1, 4, MPI_COMM_WORLD);
		MPI_Recv(park_ym, ndimer, MPI_DOUBLE, my_rank+1, 3, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				
		MPI_Send(phi, ndimer, MPI_DOUBLE, my_rank+1, 6, MPI_COMM_WORLD);
		MPI_Recv(park_phi, ndimer, MPI_DOUBLE, my_rank+1, 5, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		
		MPI_Send(dr, ndimer, MPI_DOUBLE, my_rank+1, 8, MPI_COMM_WORLD);
		MPI_Recv(park_dr, ndimer, MPI_DOUBLE, my_rank+1, 7, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		MPI_Send(&boxx, 1, MPI_DOUBLE, my_rank+1, 10, MPI_COMM_WORLD);
		MPI_Recv(&park_boxx, 1, MPI_DOUBLE, my_rank+1, 9, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		MPI_Send(&boxy, 1, MPI_DOUBLE, my_rank+1, 12, MPI_COMM_WORLD);
		MPI_Recv(&park_boxy, 1, MPI_DOUBLE, my_rank+1, 11, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		MPI_Send(&tga, 1, MPI_DOUBLE, my_rank+1, 14, MPI_COMM_WORLD);
		MPI_Recv(&park_tga, 1, MPI_DOUBLE, my_rank+1, 13, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	}
	if (my_rank==right)
	{
		if (right==0) cout <<"error rank_0"<<endl;
		
		MPI_Send(xm, ndimer, MPI_DOUBLE, my_rank-1, 1, MPI_COMM_WORLD);
		MPI_Recv(park_xm, ndimer, MPI_DOUBLE, my_rank-1, 2, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		
		MPI_Send(ym, ndimer, MPI_DOUBLE, my_rank-1, 3, MPI_COMM_WORLD);
		MPI_Recv(park_ym, ndimer, MPI_DOUBLE, my_rank-1, 4, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		
		MPI_Send(phi, ndimer, MPI_DOUBLE, my_rank-1, 5, MPI_COMM_WORLD);
		MPI_Recv(park_phi, ndimer, MPI_DOUBLE, my_rank-1, 6, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		
		MPI_Send(dr, ndimer, MPI_DOUBLE, my_rank-1, 7, MPI_COMM_WORLD);
		MPI_Recv(park_dr, ndimer, MPI_DOUBLE, my_rank-1, 8, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		MPI_Send(&boxx, 1, MPI_DOUBLE, my_rank-1, 9, MPI_COMM_WORLD);
		MPI_Recv(&park_boxx, 1, MPI_DOUBLE, my_rank-1, 10, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		MPI_Send(&boxy, 1, MPI_DOUBLE, my_rank-1, 11, MPI_COMM_WORLD);
		MPI_Recv(&park_boxy, 1, MPI_DOUBLE, my_rank-1, 12, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		MPI_Send(&tga, 1, MPI_DOUBLE, my_rank-1, 13, MPI_COMM_WORLD);
		MPI_Recv(&park_tga, 1, MPI_DOUBLE, my_rank-1, 14, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	}
	MPI_Barrier(MPI_COMM_WORLD);

// Calibrate system parameters
	if (my_rank==left || my_rank==right)
	{
	//Get values from temporary vector (park)
		for (int i=0; i<ndimer; ++i)
		{
			xm[i]=park_xm[i];
			ym[i]=park_ym[i];
			phi[i]=park_phi[i];
			dr[i]=park_dr[i];
		}
		boxx = park_boxx;
		boxy = park_boxy;
		tga = park_tga;
	// COmpute again total energy
		ene=E_Pot_Tot();
	}	
		
}

//==================
// Boltzmann_M
//==================

double Boltzmann_M (double xx, double yy, int ip)  //measure energy variation after a move
{
	int j_point, length=Neighbors[ip].size();
	double v=0.0;
    
	for (int i=0; i<length; ++i)
	{
		j_point=Neighbors[ip][i];
		v+=Int_dimer(xx, yy, phi[ip], dr[ip], xm[j_point], ym[j_point], phi[j_point], dr[j_point]);
	}
    
	return v;
}

//==================
// Boltzmann_R
//==================

double Boltzmann_R (double theta, int ip) //measure energy variation after a rotation
{
	int j_point, length=Neighbors[ip].size();
	double v=0.0;
    
	for (int i=0; i< length; ++i)
	{
		j_point=Neighbors[ip][i];
		v+=Int_dimer(xm[ip], ym[ip], theta, dr[ip], xm[j_point], ym[j_point], phi[j_point], dr[j_point]);
	}
    
	return v;
}

//==================
// Boltzmann_B
//==================

double Boltzmann_B (double dR, int ip) //measure energy variation after a break
{
	int j_point, length=Neighbors[ip].size() ;

    // Evaluate the inner energy of a dimer
	double v=V_Pot((2.*dR)*(2.*dR));			// distance=2*dR

    // Compute the new interaction energy with other dimers
	for (int i=0; i<length; ++i)
	{
		j_point=Neighbors[ip][i];
		v+=Int_dimer(xm[ip], ym[ip], phi[ip], dR, xm[j_point], ym[j_point], phi[j_point], dr[j_point]);
	}
	return v;
}

//==================
// E_Pot_Tot
//==================

double E_Pot_Tot()
{
	double v=0.0; 
	int j_point, length, i, j;

    //Loop over all the dimers
	for (i=0; i<ndimer; ++i)
		v+=V_Pot(4.*dr[i]*dr[i]); //dr è distanza tra baricentro e una delle due particelle del cluster

    //Loop over all dimer pairs
	for (i=0; i<ndimer; ++i)
	{
        length=Neighbors[i].size();
		for (j=0; j<length; ++j)
		{
			j_point=Neighbors[i][j];
			//Divide by 2 to avoid double counting
			v += Int_dimer(xm[i], ym[i], phi[i], dr[i], xm[j_point], ym[j_point], phi[j_point], dr[j_point])/2.;
		}
    }
    
    return v;
}


//==================
// Int_dimer
//==================

long double Int_dimer(double xm1, double ym1, double phi1, double dr1,
				double xm2, double ym2, double phi2, double dr2)
{
// Dimer-dimer interaction is made of 4 particle-particle interactions
// The inner interaction of each dimer is computed separately
	long double v=0.0;
	double xa, xb, xc, xd, ya, yb, yc, yd;
	double dx, dy, dR;
    double cp1=cos(phi1), cp2=cos(phi2), sp1=sin(phi1), sp2= sin(phi2);

//a
	xa = Pbcx(xm1+dr1*cp1, ym1+dr1*sp1);
	ya = Pbcy(ym1+dr1*sp1);
//b
	xb = Pbcx(xm1-dr1*cp1, ym1-dr1*sp1);
	yb = Pbcy(ym1-dr1*sp1);
//c
	xc = Pbcx(xm2+dr2*cp2, ym2+dr2*sp2);
	yc = Pbcy(ym2+dr2*sp2);
//d
	xd = Pbcx(xm2-dr2*cp2, ym2-dr2*sp2);
	yd = Pbcy(ym2-dr2*sp2);
	
//a-c
	dx = Pbcx(xa-xc, ya-yc);	
	dy = Pbcy(ya-yc); 
	dR = dx*dx+dy*dy;

	v += V_Pot(dR);
//a-d
	dx = Pbcx(xa-xd, ya-yd);	
	dy = Pbcy(ya-yd);
	dR = dx*dx+dy*dy;

	v += V_Pot(dR);
//b-c
	dx = Pbcx(xb-xc, yb-yc);	
	dy = Pbcy(yb-yc);
	dR = dx*dx+dy*dy;

	v+=V_Pot(dR);
//b-d
	dx = Pbcx(xb-xd, yb-yd);
	dy = Pbcy(yb-yd);
	dR = dx*dx+dy*dy;

	v += V_Pot(dR);

	return v;
}

//==================
// V_Pot
//==================

long double V_Pot(double r)
{
	long double v=0.0;
	if (r<rcut)
	{
		v=exp(-1.*r*r); // here r is already r^2
		if (r<rhc2)
            v+=khc;
	}
    return v;
}

//==================
// Make_Lattice
//==================

void Make_Lattice()
{
//Prepare config.
	int k=0,i,j;
	double xstart, xb, yb, theta, dist;
	double xx, yy, ratio=pi/180.;
	string nome_file; ifstream file_input;
			
	if (m_lattice==1)	//Build Lattice
	{
		if (my_rank == rank_0) cout << "Configuration generated with random angles" << endl;
		for (i=0; i<ny; ++i)
		{
			xstart = (double)i * u2x;				// first site of each lattice row

			for (j=0; j<nx; ++j)
			{
				xb = Pbcx(xstart, (double)i*u2y);
				xstart += u1x;					//increment the x of the points on that row
				yb = Pbcy((double)i*u2y);	// the points in the same row share the same y

                xm[k] = xb;
				ym[k] = yb;
				phi[k] = pi*(Rannyu()-0.5);
				dr[k++] = rhc*0.5;

			}
		}
	}
	else if (m_lattice==2)	//Read lattice from the external
	{
		if (my_rank == rank_0)
            cout << "Configuration read from file 'config.my_rank'" << endl;
		nome_file = "config."+ConvertStr(my_rank);
		const char *nome_char = nome_file.c_str();
		file_input.open(nome_char);
		for (i=0; i<ndimer; ++i)
		{
			file_input >> xb >> yb >> theta >> dist;					//read info: barycenter coordinates, orientation angle and distance between the two particles of a dimer
			xm[i] = Pbcx( xb * boxx, yb * boxy );
			ym[i] = Pbcy( yb * boxy );
			phi[i] = Pbca(theta*ratio);
			dr[i] = dist*0.5;												//each barycenter saves the distance of each of the two particles from itself
		}
		file_input.close();
	}

// Keep fixed the central dimer
	Fix_Center();
}

//==================
// Fix_Center
//==================

void Fix_Center()
{
	double xx, yy, theta, dR;
	double rr;							//Distance from the center
	double r_min=10e10;				// Thresold
	int dimer_cent;						//index of the closest dimer to the center of the lattice

//Find dimer_cent
	for (int i=0; i<ndimer; ++i)
	{
        rr = xm[i]*xm[i]+ym[i]*ym[i];
        if (rr<r_min)
        {
            dimer_cent=i;
            r_min=rr;
        }
    }
	
//Put dimer_cent at the very lattice center
	xx = xm[dimer_cent];			xm[dimer_cent] = xm[0];		xm[0] = xx;
	yy = ym[dimer_cent];			ym[dimer_cent] = ym[0];		ym[0] = yy;
	theta = phi[dimer_cent];		phi[dimer_cent] = phi[0];	phi[0] = theta;
	dR = dr[dimer_cent];			dr[dimer_cent] = dr[0];		dr[0] = dR;
}

//==================
// Def_Neighbors
//==================

void Def_Neighbors() //find neighbors of each dimer
{
	double dx, dy, dR;
    int i,j;

	Neighbors.resize(ndimer);
	
	for (i=0; i<ndimer-1; ++i)
	{
		for(j=i+1; j<ndimer; ++j)
		{
			dx = Pbcx(xm[i]-xm[j], ym[i]-ym[j]);
			dy = Pbcy(ym[i]-ym[j]);
			dR = dx*dx+dy*dy;
			if (dR<rverlet)
			{
			//update matrix
				Neighbors[i].push_back(j);
				Neighbors[j].push_back(i);
			}
		}
	}
}

//====================
// Temperature_Rate
//====================

void Temperature_Rate() //Print results for current block
{   
	ofstream file_output;
    int i;
	file_output.open("output.temperature_rate.txt");
	
	file_output<<"Move"<<endl;
	for (i=0; i<size; ++i)
		file_output <<"T= "<<1.0/beta_vec[i]<<"   E= "<<setprecision(prec)<<energy[i]<<"   E/N= "<<setprecision(prec)<<energy[i]*0.5/ndimer<<"   acceptance rate= "<<setprecision(3)<<accepted_M[i]/attemptedM<<endl;

	file_output<<"Rotate"<<endl;
	for (i=0; i<size; ++i)
		file_output <<"T= "<<1.0/beta_vec[i]<<"   E= "<<setprecision(prec)<<energy[i]<<"   E/N= "<<setprecision(prec)<<energy[i]*0.5/ndimer<<"   acceptance rate= "<<setprecision(3)<<accepted_R[i]/attemptedR<<endl;
	
	file_output<<"Break"<<endl;
	for (i=0; i<size; ++i)
		file_output <<"T= "<<1.0/beta_vec[i]<<"   E= "<<setprecision(prec)<<energy[i]<<"   E/N= "<<setprecision(prec)<<energy[i]*0.5/ndimer<<"   acceptance rate= "<<setprecision(3)<<accepted_B[i]/attemptedB<<endl;

	file_output<<"Aspect Ratio"<<endl;
	for (i=0; i<size; ++i)
		file_output <<"T= "<<1.0/beta_vec[i]<<"   E= "<<setprecision(prec)<<energy[i]<<"   E/N= "<<setprecision(prec)<<energy[i]*0.5/ndimer<<"   acceptance rate= "<<setprecision(3)<<accepted_AR[i]/attemptedAR<<endl;

	file_output<<"Angle"<<endl;
	for (i=0; i<size; ++i)
		file_output <<"T= "<<1.0/beta_vec[i]<<"   E= "<<setprecision(prec)<<energy[i]<<"   E/N= "<<setprecision(prec)<<energy[i]*0.5/ndimer<<"   acceptance rate= "<<setprecision(3)<<accepted_A[i]/attemptedA<<endl;

	file_output.close();
}

//==================
// Switch_Rate
//==================

void Switch_Rate() //Print switch rate for current block
{   
	ofstream file_output;
	file_output.open("output.switch_rate.txt");
	for (int i=0; i<size-1; ++i) 
		file_output <<"T= "<<1.0/beta_vec[i]<<" -> "<< 1.0/beta_vec[i+1] <<" = "<<accepted_S[i]/attempted_S[i]<<endl;
	file_output.close();
}

//==================
// ConvertStr
//==================

string ConvertStr (int& n)
{
    ostringstream a;
    a << n;
    return a.str();
}

//==================
// Print_Config
//==================

void Print_Config(int n, bool partial)
{
    //Create filename acording to rank and step
	ofstream file_output1, file_output2, file_output3, file_output4;
	string nome_file;
	
	if (partial)
        nome_file = "output.aconfig."+ConvertStr(my_rank)+"."+ConvertStr(n)+".txt";
	else
        nome_file = "output.aconfig."+ConvertStr(n)+".txt";
	file_output1.open(nome_file.c_str());
	
	if (partial)
        nome_file = "output.config."+ConvertStr(my_rank)+"."+ConvertStr(n)+".txt";
	else
        nome_file = "output.config."+ConvertStr(n)+".txt";
	file_output2.open(nome_file.c_str());

	if (partial)
        nome_file = "output.lattice."+ConvertStr(my_rank)+"."+ConvertStr(n)+".txt";
	else
        nome_file = "output.lattice."+ConvertStr(n)+".txt";
	file_output3.open(nome_file.c_str());	
	
	if(my_rank == rank_0)
        file_output4.open("output.MC.lattice.txt");
	for (int i=0; i<ndimer; ++i)
	{
	// use this as a new config for PT
		file_output1<< setprecision(15) <<  xm[i]/boxx<<"   "<<ym[i]/boxy<<"   "<<phi[i]*180./pi<<"   "<<2.*dr[i]<<endl;
	
	// use this as a new config for MC
		file_output2 << setprecision(15) <<  (xm[i]+dr[i]*cos(phi[i]))/boxx << "   " <<  (ym[i]+dr[i]*sin(phi[i]))/boxy << endl;
		file_output2 << setprecision(15) <<  (xm[i]-dr[i]*cos(phi[i]))/boxx << "   " <<  (ym[i]-dr[i]*sin(phi[i]))/boxy << endl;
	}

// Print lattice final info
	file_output3 << setprecision(15) <<  boxx << "    " << boxy << endl;
	file_output3 << setprecision(15) << boxx/(double) nx << endl << boxy/((double)ny*tga) << endl << boxy/(double) ny << endl << endl;

	if(my_rank == rank_0)
	{
		file_output4 << setprecision(15) << boxx/(double) nx << endl;
		file_output4 << setprecision(15) << boxy/((double)ny*tga) << endl;
		file_output4 << setprecision(15) << boxy/(double) ny << endl;
		file_output4 << nx << endl << ny << endl;
		file_output4 << "2" << endl << rho << endl;

		file_output4 << "1	ReadLattice >> u1x;		(x comp of 1st base vector)" 		<< endl;
		file_output4 << "2	ReadLattice >> u2x;		(x comp of 2nd base vector)"		<< endl;
		file_output4 << "3	ReadLattice >> u2y;		(y comp of 2nd base vector)"		<< endl;
		file_output4 << "4	ReadLattice >> nx;		(number of cells along x)"	<< endl;
		file_output4 << "5	ReadLattice >> ny;		(number of cells along y)"	<< endl;
		file_output4 << "6	ReadLattice >> ncluster"										<< endl;
		file_output4 << "7	ReadLattice >> rho"												<< endl;
	}


	file_output1.close();
	file_output2.close();
	file_output3.close();
	file_output4.close();
}

//==================
// Print_Energy
//==================

void Print_Energy(int n)
{
	ofstream file_output;
	string nome_file;
	nome_file = "output.evo."+ConvertStr(my_rank)+".txt";
	file_output.open(nome_file.c_str(), ios::app);
    file_output <<n<< "   "<<setprecision(prec)<<ene<< "     " << ene/ndimer * 0.5 << endl;
	file_output.close();
}

//==================
// Print_Box
//==================

void Print_Box(int n)
{
	ofstream file_output;
	string nome_file;
	nome_file = "output.boxevo."+ConvertStr(my_rank)+".txt";
	file_output.open(nome_file.c_str(), ios::app);
    file_output <<n<< "   "<<setprecision(prec)<< boxx/(double)nx << "     " << boxy/((double)ny*tga) << "     " << boxy/(double)ny << endl;
	file_output.close();
}

//==================
// Pbca
//==================

//Periodic Boundary Condition
double Pbca(double a)
{
    return a -  pi * rint(a/(pi));
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
// Create_Array
//==================

void Create_Array()
{
	accepted_R=new double[size];
	accepted_M=new double[size];
	accepted_B=new double[size];
	accepted_AR=new double[size];
	accepted_A=new double[size];
	accepted_S= new double[size-1];
	attempted_S= new double[size-1];
	
	beta_vec= new long double[size];
	energy=new long double[size];
}

//==================
// Delete_Array
//==================

void Delete_Array()
{
	delete [] xm;
	delete [] ym;
	delete [] phi;
	delete [] dr;
	delete [] park_xm;
	delete [] park_ym;
	delete [] park_phi;
	delete [] park_dr;
	
	if (my_rank==rank_0)
	{
		delete [] accepted_R;
		delete [] accepted_M;
		delete [] accepted_B;
		delete [] accepted_AR;
		delete [] accepted_A;
		delete [] accepted_S;
		delete [] attempted_S;
		
		delete [] beta_vec;
		delete [] energy;
	}	
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

double Rannyu()
{
	const double twom12=0.000244140625;
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
