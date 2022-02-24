#include "mpi.h"
#include <stdio.h>

int main(argc,argv)
int argc;
char **argv;
{

  int MyProc, tag=1, size;
  char msg='A', msg_recpt ;
  MPI_Status *status ;
  int left, right ;
  
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MyProc);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  

  left = (MyProc + size - 1) % size;
  right = (MyProc + 1) % size;

  if (MyProc == 0) 
  {
    printf("Proc %d sending message to proc %d \n", MyProc, right);
    
    MPI_Send(&msg, 1, MPI_CHAR, right, 1, MPI_COMM_WORLD);
    MPI_Recv(&msg_recpt, 1, MPI_CHAR, left, 1, MPI_COMM_WORLD, status);
  }
  else
  {
    MPI_Recv(&msg_recpt, 1, MPI_CHAR, left, 1, MPI_COMM_WORLD, status);
    MPI_Send(&msg, 1, MPI_CHAR, right, 1, MPI_COMM_WORLD);
    printf("Proc %d : received message from proc %d and sending message to %d\n", MyProc, left, right);
  }    

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
