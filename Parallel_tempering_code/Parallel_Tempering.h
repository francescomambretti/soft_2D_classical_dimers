#ifndef __ParallelTempering_
#define __ParallelTempering_

#include<iostream>
#include<fstream>
#include<ostream>
#include<cmath>
#include<iomanip>
#include<vector>
#include<string>
#include<sstream>
#include "mpi.h"

using namespace std;

//Pi
const double pi=3.1415927;

//Simulation Constants
int wd=20, prec=14; 
double phi_0=0.0;

//Thermodynamical state
int ndimer, nx, ny;
long double beta, temp, ene;
double rho, rcut, rverlet;
double khc, rhc, rhc2;

//Configuration scale
double u1x, u2x, u2y, tga;										// three cell base arrays and angle tangent
double boxx, boxy;
double vol;
double park_boxx, park_boxy, park_tga;

// dimer structure
double* xm; double* ym; double* phi; double* dr;
double* park_xm; double* park_ym; double* park_phi; double* park_dr;

int m_lattice;												//		-> parameter for lattice construction
bool s_lattice;												//		-> 1: all the ranks have the same configuration, 0: all different configurations

//Simulation
double alpha, delta, delta_b, delta_AR, delta_a;				//		-> delta for MC moves
int nstep, nblk, nval;											//		-> duration of MC blocks

//Parallelization
int size, my_rank; int rank_0=0;

//Accept/attempt rates
double acceptedR, attemptedR, acceptedM, attemptedM, acceptedB, attemptedB, acceptedAR, attemptedAR, acceptedA, attemptedA;
double blok_norm, ene_av, global_av, global_av2;
long double p_switch, r_switch;

double* accepted_M;
double* accepted_R;
double* accepted_B;
double* accepted_AR;
double* accepted_A;
long double* beta_vec;
long double* energy;
double* accepted_S;
double* attempted_S;

vector<vector<int> > Neighbors;

//Input Functions
void Input();												// read input file
void Make_Lattice();										// build the lattice
void Fix_Center();											// fix central dimer
void Def_Neighbors();										// define neighbors matrix

//Move Functions
void Rotate();												// rotate dimer
void Move();												// displace dimer
void Break();												// deform dimer
void Aspect_Ratio();										// change box aspect ratio
void Angle();                                               // change the angle between the two box basis vectors

void Exchange();												//check if two configurations can be switched
void Switch(int &, int &);									//and do it

//Output Functions
string ConvertStr(int& );
void Print_Config(int, bool );
void Print_Energy(int);
void Print_Box(int n);

//Energy Functions
long double V_Pot(double);										// single particle potential energy
long double Int_dimer(double xm1, double ym1, double phi1, double dr1,
				double xm2, double ym2, double phi2, double dr2);	// dimer - dimer potential energy
double Boltzmann_M(double xx, double yy, int ip);           //delta energy after dimer trial move
double Boltzmann_R(double theta, int ip);					//delta energy after dimer trial rotation
double Boltzmann_B(double dR, int ip);					//delta energy after dimer trial break
double E_Pot_Tot();											// tot. pot. energy

//Metropolis Functions
void Accumulate();
void Averages(int);
double Error(double&, double&, int &);
void Reset(int);
void Temperature_Rate();
void Switch_Rate();

void Create_Array();
void Delete_Array();

//PeriodicBoundaryConditions
double Pbca(double);									
double Pbcx(double, double);
double Pbcy(double);
#endif
