#ifndef __Random_
#define __Random_
//Random numbers generators

// random numbers
int m1,m2,m3,m4,l1,l2,l3,l4;
int n1,n2,n3,n4;

int seed[4];
void SetRandom(int *,int,int);
double Rannyu();
int igauss;
double g1,g2;
double Gauss(int,int,int,int);
void SaveRandom(int * );

#endif
