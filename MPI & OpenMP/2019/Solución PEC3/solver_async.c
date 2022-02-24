#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <mpi.h>

#define TOLERANCE 0.000001
#define MAX_ITERATIONS 20
#define N 1000

#define PRINT_MATRIX_ENABLED 0

/* Enable verbose mode */
/* #define VERBOSE */

/* Asynchronous reception of first row */
/* #define ASYNC_FIRST */

/* Asynchronous reception of last row */
#define ASYNC_LAST

#define TRUE 1
#define FALSE 0


int myRank, totalProcs;
MPI_Request* pendingRequests;
double comms_time_rows = 0, comms_time_bcast = 0;

/*
 * initializes the matrix split taking into account the
 * real absolute row number of the full global table
 */
void initialize (double **A, double **B, int rowStart, int numRows, int numCols)
{
	int i,j;

	for (i = 0; i< numRows; i++) {
		for (j = 0; j< numCols; j++) {
			A[i][0] = 0;
			B[i][0] = 0;
		}
	}
	if (rowStart == 0) {
		for (i = 0; i< numCols; i++) {
			A[0][i] = 1;
			B[0][i] = 1;
		}
	}	
	for (i = 0; i< numRows; i++) {
		A[i][0] = 1;
		B[i][0] = 1;
	}
}

/*
 * determines if the row index belongs to our split or is copied from another
 * contiguous process
 */
int is_my_row(int rowIndex, int numRows) {
	int its_mine = TRUE;
	if (myRank < totalProcs - 1 && rowIndex == numRows -1) {
		its_mine = FALSE;
	} else if (myRank > 0 && rowIndex == 0) {
		its_mine = FALSE;
	}
	return its_mine;
}

/*
 * prints the matrix in the current split
 */
void print_matrix(double **A, int firstRow, int numRows, int rowStart, int rowEnd, int numCols) {
	int i,j;

	for (i = 0; i< numRows; i++) {
		printf("[%02d] ", firstRow + i);
		for (j = 0; j< numCols; j++) {
			printf("%02.02f ", A[i][j]);
		}
		if (! is_my_row(i, numRows)) {
			printf(" (copy)");
		}
		printf("\n");
	}
	printf("\n\n");

}

/*
 * sends and receives the needed rows from/to contiguous processes
 */
void send_rows(double** A, int numRows, int numCols) {
	MPI_Status status;
	MPI_Request request;

	int tag = 1;
	double startTime = MPI_Wtime();

	if (myRank < totalProcs - 1) {
		#ifdef VERBOSE
			printf("Send row %d to %d\n", numRows - 2, myRank + 1);
		#endif
		MPI_Isend(A[numRows - 2], numCols, MPI_DOUBLE, myRank + 1, tag, MPI_COMM_WORLD, &request);

		#ifdef VERBOSE
			printf("Receive row %d from %d\n", numRows - 1, myRank + 1);
		#endif

		#ifdef ASYNC_LAST
			MPI_Irecv(A[numRows - 1], numCols, MPI_DOUBLE, myRank + 1, tag, MPI_COMM_WORLD, &pendingRequests[numRows - 1]);
		#else
			MPI_Recv(A[numRows - 1], numCols, MPI_DOUBLE, myRank + 1, tag, MPI_COMM_WORLD, &status);
		#endif
	}

	if (myRank > 0) {
		#ifdef VERBOSE
			printf("Send row 1 to %d\n", myRank - 1);
		#endif
		MPI_Isend(A[1], numCols, MPI_DOUBLE, myRank - 1, tag, MPI_COMM_WORLD, &request);

		#ifdef VERBOSE
			printf("Receive row 0 from %d\n", myRank - 1);
		#endif
		#ifdef ASYNC_FIRST
			MPI_Irecv(A[0], numCols, MPI_DOUBLE, myRank - 1, tag, MPI_COMM_WORLD, &pendingRequests[0]);
		#else
			MPI_Recv(A[0], numCols, MPI_DOUBLE, myRank - 1, tag, MPI_COMM_WORLD, &status);
		#endif
	}
	comms_time_rows += MPI_Wtime() - startTime;
}

/*
 * Determines if the full global matrix converges
 * The first process gathers the partial diff from all processes and decides
 * if in global sense the matrix has been converged. It communicates to all other
 * processes if the matrix has been converged or not to stop all processes if needed.
 */
int is_matrix_convergent(double diff, int iterationNum) {
	int convergent = FALSE,i;
	double diffProcs[totalProcs];

	double startTime = MPI_Wtime();
	MPI_Gather(&diff, 1, MPI_DOUBLE, diffProcs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	comms_time_bcast += MPI_Wtime() - startTime;

	if (myRank == 0) {
		diff = 0;
		for (i = 0; i< totalProcs; i++) {
			diff += diffProcs[i];
		}
		double ratioConvergence = diff/((double)N*(double)N);
		printf("Iteration: %d - Total diff = %.06f (ratio %.07f)\n", iterationNum, diff, ratioConvergence);

		if (ratioConvergence < TOLERANCE) {
			printf("Root convergence!\n");
			convergent=TRUE;
		}
	}

	startTime = MPI_Wtime();
	MPI_Bcast(&convergent, 1, MPI_INT, 0, MPI_COMM_WORLD);
	comms_time_bcast += MPI_Wtime() - startTime;

	return convergent;
}

/*
 * Solves the matrix iteratively
 */
void solve(double **A, double ** B, int numRows, int numCols, int firstRow, int rowStart, int rowEnd)
{
	int convergence=FALSE, iter,i;
	double diff, startTime;
	MPI_Status status;

	for (iter=0; iter < MAX_ITERATIONS; iter++) {
		#ifdef VERBOSE
			printf("[%d] Iteration: %d\n", myRank, iter);
		#endif

		diff = 0.0;

		int j;

		for (i=1; i<numRows - 1; i++) {
			#ifdef ASYNC_FIRST
				if (i == 1 && pendingRequests[i-1]) {
					startTime = MPI_Wtime();
					MPI_Wait(&pendingRequests[i-1], &status);
					comms_time_rows += MPI_Wtime() - startTime;
					pendingRequests[i-1] = 0;
				}
			#endif
			#ifdef ASYNC_LAST
				if (i == numRows - 2 && pendingRequests[i+1]) {
					startTime = MPI_Wtime();
					MPI_Wait(&pendingRequests[i+1], &status);
					comms_time_rows += MPI_Wtime() - startTime;
					pendingRequests[i+1] = 0;
				}
			#endif

			for (j=1; j<numCols - 1; j++) {
				B[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
				diff += fabs(A[i][j] - B[i][j]);
			}
		}

		/* switch matrices */
		double **p = B;
		B = A;
		A = p;

		if (PRINT_MATRIX_ENABLED) {
			print_matrix(A, firstRow, numRows, rowStart, rowEnd, N);
		}

		if (is_matrix_convergent(diff, iter)) {
			printf("R=%d convergence!\n", myRank);
			break;
		}

		send_rows(A, numRows, numCols);

		startTime = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD); //speeds up ?
		comms_time_bcast += MPI_Wtime() - startTime;
	}
}


long usecs (void)
{
	struct timeval t;

	gettimeofday(&t,NULL);
	return t.tv_sec*1000000+t.tv_usec;
}

/*
 * Returns the "result" value of the matrix.
 * It is just a sum of all the owned cells of the current
 * split without counting rows copied from contiguous processes
 */
double get_result(double **A, int numRows, int numCols)
{
	double result = 0;
	int i,j;

	for (i=0; i<numRows; i++) {
		if (is_my_row(i, numRows)) {
			for (j=0; j<numCols; j++) {
				result += A[i][j];
			}
		}
	}
	return result;
}

/*
 * Determines how many rows will have the current process
 */
void get_slice_size(int *sliceStart, int *sliceEnd, int *sliceSize) {
	int sliceWidth = (int)ceil(N / ((double)totalProcs));
	*sliceStart = sliceWidth * myRank;
	*sliceEnd = sliceWidth * (myRank + 1) - 1;
	if (myRank == totalProcs - 1) {
		*sliceEnd = N - 1;
	}
	*sliceSize = *sliceEnd - *sliceStart + 1;
}

int main(int argc, char * argv[])
{
	long t_start,t_end;
	double time;
	int i;

	t_start=usecs();

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);

	MPI_Barrier(MPI_COMM_WORLD);

	double ** A;
	double ** B;

	int rowStart, rowEnd, numRows;
	get_slice_size(&rowStart, &rowEnd, &numRows);

	/* we need to add up to 2 extra rows (rowStart -1, rowEnd + 1) */

	int allocatedNumRows;
	if (myRank == 0 || myRank == totalProcs - 1) {
		/* first and last process only borrow one row (either next or previous) */
		allocatedNumRows = numRows + 1;

	} else {
		/* other processes borrow two rows (previous, next) */
		allocatedNumRows = numRows + 2;
	}

	/* double buffering for the current split matrix */
	A = malloc((size_t)allocatedNumRows * sizeof(double *));
	B = malloc((size_t)allocatedNumRows * sizeof(double *));
	pendingRequests = malloc((size_t) allocatedNumRows * sizeof(MPI_Request));

	for (i=0; i<allocatedNumRows; i++) {
		A[i] = malloc(N * sizeof(double));
		B[i] = malloc(N * sizeof(double));
		pendingRequests[i] = 0;
	}

	int firstRow = rowStart - 1;	
	if (myRank == 0) {
		firstRow = rowStart;
	}
	initialize(A, B, firstRow, allocatedNumRows, N);

	printf("myRank: %d (of %d)\n", myRank, totalProcs);
	printf("Rows [%d -- %d] (real num rows %d)\n", rowStart, rowEnd, numRows);
	solve(A, B, allocatedNumRows, N, firstRow, rowStart, rowEnd);

	t_end=usecs();

	double procResult = get_result(A, allocatedNumRows, N);
	printf("%d - result: %.03f\n", myRank, procResult);

	double* results = malloc(sizeof(double) * totalProcs);

	double startTime = MPI_Wtime();
	MPI_Gather(&procResult, 1, MPI_DOUBLE, results, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	comms_time_bcast += MPI_Wtime() - startTime;

	if (myRank == 0) {
		double totalResult = 0;
		for ( i = 0; i< totalProcs; i++) {
			totalResult += results[i];
		}
		printf("Total result: %.03f\n", totalResult);
	}
	free(results);
	free(A);
	free(B);

	startTime = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	comms_time_bcast += MPI_Wtime() - startTime;

	MPI_Finalize();

	time = ((double)(t_end-t_start))/1000000;
	printf("[%d] total time = %.03f s\n", myRank, time);
	printf("[%d] computation time = %.03f s\n", myRank, time - comms_time_rows);
	printf("[%d] communications time rows = %.03f s, barriers = %.03f\n", myRank, comms_time_rows, comms_time_bcast);

	double computation_time = time - comms_time_bcast - comms_time_rows;
	printf("computation,%d,%.03f,%.03f,%.03f\n", myRank, computation_time, comms_time_rows, comms_time_bcast);

	return(0);
}

