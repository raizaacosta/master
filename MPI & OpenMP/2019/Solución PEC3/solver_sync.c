
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h> 
#include "mpi.h"

#define Tolerance 0.00001
#define TRUE 1
#define FALSE 0

#define N 100

#define ITERS 20

double ** A;
int rank, numprocs, tag=1;
MPI_Status status;

int initialize (double **A, int n)
{
   int i,j;

   for (j=0;j<n+1;j++){
     A[0][j]=1.0;
   }
   for (i=1;i<n+1;i++){
      A[i][0]=1.0;
      for (j=1;j<n+1;j++) A[i][j]=0.0;
   }

}

void solve(double **A, int n)
{
   int convergence=FALSE;
   double diff, tmp;
   int i,j, iters=0;
   int for_iters;
   int fila1, filaFin, numfilas;

	 /*determino el numero de filas que procesa cada proceso, asi como la fila primera y la fila fin de cada proceso*/
	 numfilas = (int) (N+2) / numprocs; 
	 fila1 = numfilas * rank ;
	 filaFin = fila1 + numfilas;
	 
	 /*printf("Proceso # %d/%d numfilas %d  fila1 %d filaFin %d\n", rank,numprocs, numfilas, fila1, filaFin);*/
	 
	 /* cuando el rango es 0 la fila1 seria la 0 como esa no queremos procesarla cambio el valor de fila1 a 1*/
	 if (rank == 0) fila1 = 1;
	 /* cuando el rango es igual al numero de procesos -1 modifico el valor de la filaFIn para que se corresponda con N+1*/
	 /* para procesar hasta la fila N+1 y de este modo no procesar la fila N+2*/
	 if (rank == numprocs-1) filaFin = N+1;
	 
	 /*printf("Proceso # %d/%d numfilas %d  fila1 %d filaFin %d\n", rank,numprocs, numfilas, fila1, filaFin);*/


   for (for_iters=1;for_iters<21;for_iters++) 
   { 
	 
	 diff = 0.0;

	/*en este punto se envian y se reciben los mensajes*/	 
	 if (rank == 0) /*se envia la última fila al proceso 1 y recibe la fila siguiente del proceso 1*/
	 {
		 /*printf("El proceso # %d Esta enviando mensaje de la filaFin-1 A[%d] al proceso #%d  \n", rank, filaFin-1, rank + 1) ;*/
		 MPI_Send(A[filaFin-1], N+2, MPI_DOUBLE, rank + 1, tag, MPI_COMM_WORLD );
		 /*printf("El proceso # %d Ha enviado    mensaje de la filaFin-1 A[%d] al proceso #%d  \n", rank, filaFin-1, rank + 1) ;*/

		 /*printf("El proceso # %d esta Recibiendo mensaje de la filaFin A[%d] del proceso #%d   \n", rank , filaFin, rank + 1 ) ;*/
		 MPI_Recv( A[filaFin],  N+2, MPI_DOUBLE, rank + 1, tag, MPI_COMM_WORLD, &status );
		 /*printf("El proceso # %d Ha recibido mensaje     de la filaFin A[%d] del proceso #%d   \n", rank , filaFin, rank + 1 ) ;*/
	 }
	 else if (rank == numprocs-1) /*se envia la primera fila al proceso anterior (rank - 1) y recibe la fila anterior del proceso (rank - 1)*/
	 {
		 /*printf("El proceso # %d Esta Recibiendo mensaje de la fila1 A[%d] del proceso #%d   \n", rank, fila1-1, rank -1) ;*/
		 MPI_Recv( A[fila1 -1],  N+2, MPI_DOUBLE, rank - 1, tag, MPI_COMM_WORLD, &status );
		 /*printf("El proceso # %d Ha recibido mensaje     de la fila1 A[%d] del proceso #%d  \n \n", rank, fila1-1, rank -1) ;*/

		 /*printf("El proceso # %d Esta enviando mensaje de la fila1 A[%d] al proceso #%d   \n", rank, fila1, rank - 1) ;*/
		 MPI_Send(A[fila1], N+2, MPI_DOUBLE, rank - 1, tag, MPI_COMM_WORLD );
		 /*printf("El proceso # %d ha enviado    mensaje de la fila1 A[%d] al proceso #%d   \n\n", rank, fila1, rank - 1) ;*/
	 }
	 else /* Recibe la ultima fila del proceso anterior y  envia l a primera fila al  proceso anterior*/
	 {    /* Envia  la ultima fila al  proceso siguiente y recibe la primera fila del proceso siguiente*/
	 
		 /*printf("El proceso # %d Esta Recibiendo mensaje de la fila1 A[%d] del proceso #%d   \n", rank, fila1 -1, rank -1) ;*/
		 MPI_Recv( A[fila1-1],  N+2, MPI_DOUBLE, rank - 1, tag, MPI_COMM_WORLD, &status );
		 /*printf("El proceso # %d ha recibido mensaje     de la fila1 A[%d] del proceso #%d   \n\n", rank, fila1 -1, rank -1) ;*/

		 /*printf("El proceso # %d Esta enviando mensaje de la fila1 A[%d] al proceso #%d   \n", rank, fila1, rank - 1) ;*/
		 MPI_Send(A[fila1], N+2, MPI_DOUBLE, rank - 1, tag, MPI_COMM_WORLD );
		 /*printf("El proceso # %d ha enviado    mensaje de la fila1 A[%d] al proceso #%d   \n\n", rank, fila1, rank - 1) ;*/

		 /*printf("El proceso # %d Esta enviando mensaje de la filaFin-1 A[%d] al proceso #%d   \n", rank, filaFin-1, rank + 1) ;*/
		 MPI_Send(A[filaFin-1], N+2, MPI_DOUBLE, rank + 1, tag, MPI_COMM_WORLD );
		 /*printf("El proceso # %d ha enviado    mensaje de la filaFin-1 A[%d] al proceso #%d  \n \n", rank, filaFin-1, rank + 1) ;*/

		 /*printf("El proceso # %d Esta Recibiendo mensaje de la filaFin A[%d] del proceso #%d  \n", rank , filaFin, rank +1) ;*/
		 MPI_Recv( A[filaFin],  N+2, MPI_DOUBLE, rank + 1, tag, MPI_COMM_WORLD, &status );
		 /*printf("El proceso # %d Ha recibido mensaje     de la filaFin A[%d] del proceso #%d  \n\n", rank , filaFin, rank +1) ;*/
	 }
	
     for (i=fila1;i<filaFin;i++)
     {
		
		for (j=1;j<n;j++)
		{
			tmp = A[i][j];
			A[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
			diff += fabs(A[i][j] - tmp);
         /*printf("partial dif is %f \n ", A[i][j] - tmp);*/
		}
		printf("\n");
		printf("---------------------------------------");
		printf("\n");
		iters++;
		printf("%f %f\n", diff, Tolerance);
		if (diff/((double)N*(double)N) < Tolerance){
			convergence=TRUE;
			printf("%u\n", convergence);
		}
 	 }
      

    } /*for*/

    printf("%u\n", convergence);
}


long usecs (void)
{
  struct timeval t;

  gettimeofday(&t,NULL);
  return t.tv_sec*1000000+t.tv_usec;
}


int main(int argc, char * argv[])
{
  int i;
  long t_start,t_end;
  double time;
  char hostname[256];
 

  A = malloc((N+2) * sizeof(double *));
  for (i=0; i<N+2; i++) {
    A[i] = malloc((N+2) * sizeof(double)); 
  }

  initialize(A, N);
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  /* proceso que se esta ejecutando */
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);  /* numero de procesos */
  gethostname(hostname,255); /* cola en la que se esta ejecutando */

  printf("\nProceso # %d/%d iniciado en %s\n", rank,numprocs, hostname);
 
  
  t_start=usecs();
  
  solve(A, N);
  
  t_end=usecs();
  
  MPI_Finalize();

  time = ((double)(t_end-t_start))/1000000;
  printf("Computation time = %f\n", time);
  
  return 0;
}

