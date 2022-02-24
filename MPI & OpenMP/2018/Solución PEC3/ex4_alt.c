#include "mpi.h"
#include <stdio.h>

int main(argc,argv)
int argc;
char **argv;
{

	int current, total, i, tag=1;
	char msg='A', msg_recpt;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Comm_rank(MPI_COMM_WORLD, &current);

	printf("Process # %d started \n", current);
	MPI_Barrier(MPI_COMM_WORLD);
	

	for(i = 0; i<total ; i++){		
		if(i < current){
			MPI_Recv(&msg_recpt, 1, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
			printf("Proc #%d received message from Proc #%d\n", current, i) ;
		}
		if(i > current){
			printf("Proc #%d sending message to Proc #%d\n", current, i) ;
			MPI_Send(&msg, 1, MPI_CHAR, i, tag, MPI_COMM_WORLD);
		}		
	}
	
	for(i = 0; i<total ; i++){		
		if(i < current){
			printf("Proc #%d sending message to Proc #%d\n", current, i) ;
			MPI_Send(&msg, 1, MPI_CHAR, i, tag, MPI_COMM_WORLD);
		}
		if(i > current){
			MPI_Recv(&msg_recpt, 1, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
			printf("Proc #%d received message from Proc #%d\n", current, i) ;
		}		
	}
	

	
	printf("Finishing proc %d\n", current); 

	MPI_Barrier(MPI_COMM_WORLD); 
	MPI_Finalize();
}