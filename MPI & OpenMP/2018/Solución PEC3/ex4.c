#include <mpi.h>
#include <stdio.h>

int main(argc,argv)
int argc;
char **argv;
{

	int MyProc, tag=1, nProcs, i;
	char msg='A', msg_recpt;
	MPI_Request request;
	MPI_Status status;
  
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyProc);
   
	// Getting number of threads
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
	
	// Sending a message to the rest of threads
	for (i=0; i<nProcs; i++) {
		
		// Excluding sending the message to itself
		if (MyProc != i) {
			printf("Proc #%d sending message to Proc #%d\n", MyProc, i);
			MPI_Isend(&msg, 1, MPI_CHAR, i, tag, MPI_COMM_WORLD, &request);		
		}
	}
	
	// Receiving a message from the rest of threads
	for (i=0; i<nProcs; i++) {
		
		// Excluding receiving the message from itself
		if (MyProc != i) {
			printf("Proc #%d received message from Proc #%d\n", MyProc, i);
			MPI_Irecv(&msg_recpt, 1, MPI_CHAR, i, tag, MPI_COMM_WORLD, &request);		
		}
	}
    
    //MPI_Barrier or waits can be considered
	MPI_Finalize();
	
	return 0;
}

