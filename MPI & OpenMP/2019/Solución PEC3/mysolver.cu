#include <stdio.h>
#include "cuPrintf.cu"
#define N 10 
#define M 12
#define ITERS 100

__global__
void mysolver(float *a, float *b)
{
 int columna = threadIdx.x; // indice de columna
 int fila = threadIdx.y; // indice de fila
 int myID = columna + fila * blockDim.x; //indice lineal
 int arriba=myID-M;
 int abajo=myID+M;
 int derecha=myID+1;
 int izquierda=myID-1;

  if (! ( (myID < M) || (myID % M == 0) || (myID > M*(M-1)) || ((myID+1) %M == 0) )) {
   b[myID] = 0.2 *(a[myID]+a[arriba]+a[abajo]+a[derecha]+a[izquierda]);}
}


int main()
{
float *hst_A, *hst_B;
float *dev_A, *dev_B;

hst_A = (float*)malloc(M*M*sizeof(float));
hst_B = (float*)malloc(M*M*sizeof(float));


cudaMalloc( (void**)&dev_A, M*M*sizeof(float));
cudaMalloc( (void**)&dev_B, M*M*sizeof(float));

for(int i=0;i<M*M;i++)
{
 if ( (i < M) || (i % M == 0) || (i > M*(M-1)) || ((i+1) %M == 0) )
 {
  hst_A[i] = 0;}
 else {
  //hst_A[i]= (float)( rand() % 10 );}
  hst_A[i]= 100;
 }
} 
 
 
cudaMemcpy( dev_A, hst_A, M*M*sizeof(float), cudaMemcpyHostToDevice );
cudaMemcpy( dev_B, hst_B, M*M*sizeof(float), cudaMemcpyHostToDevice );


// dimensiones del kernel
dim3 Nbloques(1);
dim3 hilosB(M,M);

for(int i=0;i<ITERS;i++){
 mysolver<<<Nbloques,hilosB>>>(dev_A, dev_B);
 cudaMemcpy( dev_A, dev_B, M*M*sizeof(float), cudaMemcpyDeviceToDevice );
}

cudaMemcpy( hst_B, dev_B, M*M*sizeof(float), cudaMemcpyDeviceToHost );
cudaMemcpy( hst_A, dev_A, M*M*sizeof(float), cudaMemcpyDeviceToHost );

printf("B:\n");
for(int i=0;i<M;i++)
{
for(int j=0;j<M;j++)
{
printf("%0.2f ",hst_B[j+i*M]);
}
printf("\n");
}


printf("A:\n");
for(int i=0;i<M;i++)
{
for(int j=0;j<M;j++)
{
printf("%0.2f ",hst_A[j+i*M]);
}
printf("\n");
}

return EXIT_SUCCESS;
}
