#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <arm_sve.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */
void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_transpose( int ii, int jj, int kk, float *A, float *B, float *C, float ALPHA, int M, int N, int K,  int lda,int ldb,int ldc)
{

//int ii =0, jj=0, kk=0;
// #pragma omp parallel //private(M)
//        {
//int i, j, k;
int i=ii, j=jj, k=kk;
//int num = omp_get_num_threads(void);
//printf("%d num of threads ", num);
//M = M/2;
// #pragma omp for 
for( j = jj; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
        for (i = ii; i < M-15; i += 16) {
        svfloat32_t vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, vc16, vc17, vc18, vc19, vc20, vc21, vc22, vc23;// vc24, vc25, vc26, vc27, vc28, vc29, vc30, vc31;
        vc= svld1(pg, &C[i*ldc+j]);
        vc1= svld1(pg, &C[(i+1)*ldc+j]);
        vc2= svld1(pg, &C[(i+2)*ldc+j]);
        vc3= svld1(pg, &C[(i+3)*ldc+j]);
        vc4= svld1(pg, &C[(i+4)*ldc+j]);
        vc5= svld1(pg, &C[(i+5)*ldc+j]);
        vc6= svld1(pg, &C[(i+6)*ldc+j]);
        vc7= svld1(pg, &C[(i+7)*ldc+j]);
        vc8= svld1(pg, &C[(i+8)*ldc+j]);
        vc9= svld1(pg, &C[(i+9)*ldc+j]);
        vc10= svld1(pg, &C[(i+10)*ldc+j]);
        vc11= svld1(pg, &C[(i+11)*ldc+j]);
        vc12= svld1(pg, &C[(i+12)*ldc+j]);
        vc13= svld1(pg, &C[(i+13)*ldc+j]);
        vc14= svld1(pg, &C[(i+14)*ldc+j]);
        vc15= svld1(pg, &C[(i+15)*ldc+j]);
        svfloat32_t vb,vb1,vb2,vb3,vb4,vb5,vb6,vb7;

	        int flag =0;
        for ( k = kk; k < K; k += 1) {
                __builtin_prefetch(B, 0, 3);
                  svfloat32_t vb = svld1(pg, &B[k*ldb+j]);

                __builtin_prefetch(A, 0, 3);
                register float alpha =  A[i+lda*k];
                register float alpha1 =  A[(i+1)+lda*k];
                register float alpha2 =  A[(i+2)+lda*k];
                register float alpha3 =  A[(i+3)+lda*k];
                register float alpha4 =  A[(i+4)+lda*k];
                register float alpha5 =  A[(i+5)+lda*k];
                register float alpha6 =  A[(i+6)+lda*k];
                register float alpha7 =  A[(i+7)+lda*k];
                register float alpha8 =  A[(i+8)+lda*k];
                register float alpha9 =  A[(i+9)+lda*k];
                register float alpha10 =  A[(i+10)+lda*k];
                 register float alpha11 =  A[(i+11)+lda*k];
                register float alpha12 =  A[(i+12)+lda*k];
                register float alpha13 =  A[(i+13)+lda*k];
                register float alpha14 =  A[(i+14)+lda*k];
                register float alpha15 =  A[(i+15)+lda*k];
                  vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
           vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B
        }
                svst1(pg, &C[i*ldc+j], vc);
                svst1(pg, &C[(i+1)*ldc+j], vc1);
                svst1(pg, &C[(i+2)*ldc+j], vc2);
                svst1(pg, &C[(i+3)*ldc+j], vc3);
                svst1(pg, &C[(i+4)*ldc+j], vc4);
                svst1(pg, &C[(i+5)*ldc+j], vc5);
                svst1(pg, &C[(i+6)*ldc+j], vc6);
                svst1(pg, &C[(i+7)*ldc+j], vc7);
                svst1(pg, &C[(i+8)*ldc+j], vc8);
                svst1(pg, &C[(i+9)*ldc+j], vc9);
                svst1(pg, &C[(i+10)*ldc+j], vc10);
                svst1(pg, &C[(i+11)*ldc+j], vc11);
                svst1(pg, &C[(i+12)*ldc+j], vc12);
                svst1(pg, &C[(i+13)*ldc+j], vc13);
                svst1(pg, &C[(i+14)*ldc+j], vc14);
                svst1(pg, &C[(i+15)*ldc+j], vc15);
        }
        }//}

//int k_left = k;
  int i_left=i;
  //  #pragma omp for
  for ( j = jj; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
     svfloat32_t  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     for (i=i_left; i < M; i += 4) {
        vc= svld1(pg, &C[i*ldc+j]);
        if((i+1) < M)  {vc1= svld1(pg, &C[(i+1)*ldc+j]);}
        if ((i+2) < M) {vc2= svld1(pg, &C[(i+2)*ldc+j]);}
        if ((i+3) < M) {vc3= svld1(pg, &C[(i+3)*ldc+j]);}
       for (int k = kk; k < K; k += 1) {
                alpha =  A[i+lda*k];
                if ((i+1) < M) {alpha1 =  A[(i+1)+lda*k]; }
                if ((i+2) < M) { alpha2 =  A[(i+2)+lda*k];}
                if ((i+3) < M) { alpha3 =  A[(i+3)+lda*k];}
                vb = svld1(pg, &B[k*ldb+j]);
                  vc = svmla_m(pg, vc, vb, alpha); // sum += ALPHA*A*B
                  if ((i+1) < M) {vc1 = svmla_m(pg, vc1, vb, alpha1);} // sum += ALPHA*A*B
                  if ((i+2) < M) {vc2 = svmla_m(pg, vc2, vb, alpha2);} // sum += ALPHA*A*B
                  if ((i+3) < M) {vc3 = svmla_m(pg, vc3, vb, alpha3);}// sum += ALPHA*A*B
        }
          svst1(pg, &C[i*ldc+j], vc);
          if ((i+1) < M)      {svst1(pg, &C[(i+1)*ldc+j], vc1);}
          if ((i+2) < M)      {svst1(pg, &C[(i+2)*ldc+j], vc2);}
          if ((i+3) < M)      {svst1(pg, &C[(i+3)*ldc+j], vc3);}
     }
  }//}
}


void gemm_nn1_transpose(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, float *transpose,  int ldc, int BlockM, int BlockN, int BlockK)
{       
        //int BlockM = 64, BlockN=1024, BlockK=256;
       // int BlockM = 32, BlockN=2048, BlockK=256;
 //       int BlockM = 128, BlockN=2048, BlockK=256;

for(int i=0;i<M;i++)
{       
        for(int k=0;k<K;k++)
        {       
                transpose[k*M+i] = A[i*lda+k];
        }
}
        
        int ii,jj,kk,i,j,k;
 for (jj = 0; jj < N; jj+=BlockN) {
    for (kk = 0; kk < K; kk+=BlockK) {
        for (ii = 0; ii < M; ii+=BlockM) {
        int Mc = ((ii+BlockM >M)?M:(ii+BlockM)) ;
        int Nc = ((jj+BlockN>N)?N:(jj+BlockN));
        int Kc = ((kk+BlockK > K)?K:(kk+BlockK));
        
        gemm_transpose(ii,jj,kk,transpose,B, C,ALPHA, Mc,Nc,Kc, M,ldb,ldc );

	}
	}
}
}


void gemm_new1db( int ii, int jj, int kk, float *A, float *B, float *C, float ALPHA, int M, int N, int K,  int lda,int ldb,int ldc)
{
 #pragma omp parallel
{
		int itr =0;
int i, j, k;
int i1=ii, j1=jj, k1=kk;
//int num = omp_get_num_threads(void);
//printf("%d num of threads ", num);
//M = M/2;
#pragma omp for 
for( j = 0; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
        for (i = 0; i < M-15; i += 16) {
                __builtin_prefetch(&C[(i+i1)*ldc+(j+j1)], 0, 3);
                __builtin_prefetch(B, 0, 2);
                __builtin_prefetch(A, 0, 2);
        svfloat32_t vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, vc16, vc17, vc18, vc19, vc20, vc21, vc22, vc23;// vc24, vc25, vc26, vc27, vc28, vc29, vc30, vc31;
        vc= svld1(pg, &C[(i+i1)*ldc+(j+j1)]);
        vc1= svld1(pg, &C[(i+i1+1)*ldc+(j+j1)]);
        vc2= svld1(pg, &C[(i+i1+2)*ldc+(j+j1)]);
        vc3= svld1(pg, &C[(i+i1+3)*ldc+(j+j1)]);
        vc4= svld1(pg, &C[(i+i1+4)*ldc+(j+j1)]);
        vc5= svld1(pg, &C[(i+i1+5)*ldc+(j+j1)]);
        vc6= svld1(pg, &C[(i+i1+6)*ldc+(j+j1)]);
        vc7= svld1(pg, &C[(i+i1+7)*ldc+(j+j1)]);
        vc8= svld1(pg, &C[(i+i1+8)*ldc+(j+j1)]);
        vc9= svld1(pg, &C[(i+i1+9)*ldc+(j+j1)]);
        vc10= svld1(pg, &C[(i+i1+10)*ldc+(j+j1)]);
        vc11= svld1(pg, &C[(i+i1+11)*ldc+(j+j1)]);
        vc12= svld1(pg, &C[(i+i1+12)*ldc+(j+j1)]);
        vc13= svld1(pg, &C[(i+i1+13)*ldc+(j+j1)]);
        vc14= svld1(pg, &C[(i+i1+14)*ldc+(j+j1)]);
        vc15= svld1(pg, &C[(i+i1+15)*ldc+(j+j1)]);
                int flag =0;
                __builtin_prefetch(B, 0, 3);
                __builtin_prefetch(A, 0, 3);
		 //#pragma omp for 
    	svfloat32_t vb,vb1,vb2,vb3,vb4,vb5,vb6,vb7;
        for ( k = 0; k < K-3; k += 4) {
		if (flag==0){
		 vb = svld1(pg, &B[((k+(K*(j/ldb)))*ldb)+0]);
		vb1 = svld1(pg, &B[(((k+1)+(K*(j/ldb)))*ldb)+0]);
		vb2 = svld1(pg, &B[(((k+2)+(K*(j/ldb)))*ldb)+0]);
		vb3 = svld1(pg, &B[(((k+3)+(K*(j/ldb)))*ldb)+0]);


		 vb4 = svld1(pg, &B[(((k+4)+(K*(j/ldb)))*ldb)+0]);
                vb5 = svld1(pg, &B[(((k+5)+(K*(j/ldb)))*ldb)+0]);
                vb6 = svld1(pg, &B[(((k+6)+(K*(j/ldb)))*ldb)+0]);
                vb7 = svld1(pg, &B[(((k+7)+(K*(j/ldb)))*ldb)+0]);


		}
		else
		{
			if(flag & 1)	//odd number
			{
				if(k<K-4)
				{
                                        vb = svld1(pg, &B[(((k+4)+(K*(j/ldb)))*ldb)+0]);
                                        vb1 = svld1(pg, &B[(((k+5)+(K*(j/ldb)))*ldb)+0]);
                                        vb2 = svld1(pg, &B[(((k+6)+(K*(j/ldb)))*ldb)+0]);
                                        vb3 = svld1(pg, &B[(((k+7)+(K*(j/ldb)))*ldb)+0]);
				}
			}			
			else  //even number
			{
				if(k<K-4)
                                {
					vb4 = svld1(pg, &B[(((k+4)+(K*(j/ldb)))*ldb)+0]);
                			vb5 = svld1(pg, &B[(((k+5)+(K*(j/ldb)))*ldb)+0]);
                			vb6 = svld1(pg, &B[(((k+6)+(K*(j/ldb)))*ldb)+0]);
                			vb7 = svld1(pg, &B[(((k+7)+(K*(j/ldb)))*ldb)+0]);
                                }
			}
		}
		if(flag & 1)
		{
                register float alpha =  A[i+lda*k];
                register float alpha1 =  A[(i+1)+lda*k];
                register float alpha2 =  A[(i+2)+lda*k];
                register float alpha3 =  A[(i+3)+lda*k];
                register float alpha4 =  A[(i+4)+lda*k];
                register float alpha5 =  A[(i+5)+lda*k];
                register float alpha6 =  A[(i+6)+lda*k];
                register float alpha7 =  A[(i+7)+lda*k];
                register float alpha8 =  A[(i+8)+lda*k];
                register float alpha9 =  A[(i+9)+lda*k];
                register float alpha10 =  A[(i+10)+lda*k];
                 register float alpha11 =  A[(i+11)+lda*k];
                register float alpha12 =  A[(i+12)+lda*k];
                register float alpha13 =  A[(i+13)+lda*k];
                register float alpha14 =  A[(i+14)+lda*k];
                register float alpha15 =  A[(i+15)+lda*k];
                  vc = svmla_m(pg,vc, vb4, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb4, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb4, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb4, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb4, alpha4); // sum += ALPHA*A*B
           	vc5 = svmla_m(pg, vc5, vb4, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb4, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb4, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb4, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb4, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb4, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb4, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb4, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb4, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb4, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb4, alpha15); // sum += ALPHA*A*B
                alpha =  A[i+lda*(k+1)];
                 alpha1 =  A[(i+1)+lda*(k+1)];
                 alpha2 =  A[(i+2)+lda*(k+1)];
                 alpha3 =  A[(i+3)+lda*(k+1)];
                 alpha4 =  A[(i+4)+lda*(k+1)];
                 alpha5 =  A[(i+5)+lda*(k+1)];
                 alpha6 =  A[(i+6)+lda*(k+1)];
                 alpha7 =  A[(i+7)+lda*(k+1)];
                 alpha8 =  A[(i+8)+lda*(k+1)];
                 alpha9 =  A[(i+9)+lda*(k+1)];
                 alpha10 =  A[(i+10)+lda*(k+1)];
                  alpha11 =  A[(i+11)+lda*(k+1)];
                 alpha12 =  A[(i+12)+lda*(k+1)];
                 alpha13 =  A[(i+13)+lda*(k+1)];
                 alpha14 =  A[(i+14)+lda*(k+1)];
                 alpha15 =  A[(i+15)+lda*(k+1)];
                  vc = svmla_m(pg,vc, vb5, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb5, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb5, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb5, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb5, alpha4); // sum += ALPHA*A*B
           	vc5 = svmla_m(pg, vc5, vb5,alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb5, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb5, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb5, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb5, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb5, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb5, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb5, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb5, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb5, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb5, alpha15); // sum += ALPHA*A*B
                alpha =  A[i+lda*(k+2)];
                 alpha1 =  A[(i+1)+lda*(k+2)];
                 alpha2 =  A[(i+2)+lda*(k+2)];
                 alpha3 =  A[(i+3)+lda*(k+2)];
                 alpha4 =  A[(i+4)+lda*(k+2)];
                 alpha5 =  A[(i+5)+lda*(k+2)];
                 alpha6 =  A[(i+6)+lda*(k+2)];
                 alpha7 =  A[(i+7)+lda*(k+2)];
                 alpha8 =  A[(i+8)+lda*(k+2)];
                 alpha9 =  A[(i+9)+lda*(k+2)];
                 alpha10 =  A[(i+10)+lda*(k+2)];
                  alpha11 =  A[(i+11)+lda*(k+2)];
                 alpha12 =  A[(i+12)+lda*(k+2)];
                 alpha13 =  A[(i+13)+lda*(k+2)];
                 alpha14 =  A[(i+14)+lda*(k+2)];
                 alpha15 =  A[(i+15)+lda*(k+2)];
                  vc = svmla_m(pg,vc, vb6, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb6, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb6, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb6, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb6, alpha4); // sum += ALPHA*A*B
           	vc5 = svmla_m(pg, vc5, vb6,alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb6, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb6, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb6, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb6, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb6, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb6, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb6, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb6, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb6, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb6, alpha15); // sum += ALPHA*A*B
                alpha =  A[i+lda*(k+3)];
                 alpha1 =  A[(i+1)+lda*(k+3)];
                 alpha2 =  A[(i+2)+lda*(k+3)];
                 alpha3 =  A[(i+3)+lda*(k+3)];
                 alpha4 =  A[(i+4)+lda*(k+3)];
                 alpha5 =  A[(i+5)+lda*(k+3)];
                 alpha6 =  A[(i+6)+lda*(k+3)];
                 alpha7 =  A[(i+7)+lda*(k+3)];
                 alpha8 =  A[(i+8)+lda*(k+3)];
                 alpha9 =  A[(i+9)+lda*(k+3)];
                 alpha10 =  A[(i+10)+lda*(k+3)];
                  alpha11 =  A[(i+11)+lda*(k+3)];
                 alpha12 =  A[(i+12)+lda*(k+3)];
                 alpha13 =  A[(i+13)+lda*(k+3)];
                 alpha14 =  A[(i+14)+lda*(k+3)];
                 alpha15 =  A[(i+15)+lda*(k+3)];
                  vc = svmla_m(pg,vc, vb7, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb7, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb7, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb7, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb7, alpha4); // sum += ALPHA*A*B
           	vc5 = svmla_m(pg, vc5, vb7,alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb7, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb7, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb7, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb7, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb7, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb7, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb7, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb7, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb7, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb7, alpha15); // sum += ALPHA*A*B
		}
		else
		{
                register float alpha =  A[i+lda*k];
                register float alpha1 =  A[(i+1)+lda*k];
                register float alpha2 =  A[(i+2)+lda*k];
                register float alpha3 =  A[(i+3)+lda*k];
                register float alpha4 =  A[(i+4)+lda*k];
                register float alpha5 =  A[(i+5)+lda*k];
                register float alpha6 =  A[(i+6)+lda*k];
                register float alpha7 =  A[(i+7)+lda*k];
                register float alpha8 =  A[(i+8)+lda*k];
                register float alpha9 =  A[(i+9)+lda*k];
                register float alpha10 =  A[(i+10)+lda*k];
                 register float alpha11 =  A[(i+11)+lda*k];
                register float alpha12 =  A[(i+12)+lda*k];
                register float alpha13 =  A[(i+13)+lda*k];
                register float alpha14 =  A[(i+14)+lda*k];
                register float alpha15 =  A[(i+15)+lda*k];
                  vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
           	vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B
                alpha =  A[i+lda*(k+1)];
                 alpha1 =  A[(i+1)+lda*(k+1)];
                 alpha2 =  A[(i+2)+lda*(k+1)];
                 alpha3 =  A[(i+3)+lda*(k+1)];
                 alpha4 =  A[(i+4)+lda*(k+1)];
                 alpha5 =  A[(i+5)+lda*(k+1)];
                 alpha6 =  A[(i+6)+lda*(k+1)];
                 alpha7 =  A[(i+7)+lda*(k+1)];
                 alpha8 =  A[(i+8)+lda*(k+1)];
                 alpha9 =  A[(i+9)+lda*(k+1)];
                 alpha10 =  A[(i+10)+lda*(k+1)];
                  alpha11 =  A[(i+11)+lda*(k+1)];
                 alpha12 =  A[(i+12)+lda*(k+1)];
                 alpha13 =  A[(i+13)+lda*(k+1)];
                 alpha14 =  A[(i+14)+lda*(k+1)];
                 alpha15 =  A[(i+15)+lda*(k+1)];
                  vc = svmla_m(pg,vc, vb1, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb1, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb1, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb1, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb1, alpha4); // sum += ALPHA*A*B
           	vc5 = svmla_m(pg, vc5, vb1,alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb1, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb1, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb1, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb1, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb1, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb1, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb1, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb1, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb1, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb1, alpha15); // sum += ALPHA*A*B
                alpha =  A[i+lda*(k+2)];
                 alpha1 =  A[(i+1)+lda*(k+2)];
                 alpha2 =  A[(i+2)+lda*(k+2)];
                 alpha3 =  A[(i+3)+lda*(k+2)];
                 alpha4 =  A[(i+4)+lda*(k+2)];
                 alpha5 =  A[(i+5)+lda*(k+2)];
                 alpha6 =  A[(i+6)+lda*(k+2)];
                 alpha7 =  A[(i+7)+lda*(k+2)];
                 alpha8 =  A[(i+8)+lda*(k+2)];
                 alpha9 =  A[(i+9)+lda*(k+2)];
                 alpha10 =  A[(i+10)+lda*(k+2)];
                  alpha11 =  A[(i+11)+lda*(k+2)];
                 alpha12 =  A[(i+12)+lda*(k+2)];
                 alpha13 =  A[(i+13)+lda*(k+2)];
                 alpha14 =  A[(i+14)+lda*(k+2)];
                 alpha15 =  A[(i+15)+lda*(k+2)];
                  vc = svmla_m(pg,vc, vb2, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb2, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb2, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb2, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb2, alpha4); // sum += ALPHA*A*B
           	vc5 = svmla_m(pg, vc5, vb2,alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb2, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb2, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb2, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb2, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb2, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb2, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb2, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb2, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb2, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb2, alpha15); // sum += ALPHA*A*B
                alpha =  A[i+lda*(k+3)];
                 alpha1 =  A[(i+1)+lda*(k+3)];
                 alpha2 =  A[(i+2)+lda*(k+3)];
                 alpha3 =  A[(i+3)+lda*(k+3)];
                 alpha4 =  A[(i+4)+lda*(k+3)];
                 alpha5 =  A[(i+5)+lda*(k+3)];
                 alpha6 =  A[(i+6)+lda*(k+3)];
                 alpha7 =  A[(i+7)+lda*(k+3)];
                 alpha8 =  A[(i+8)+lda*(k+3)];
                 alpha9 =  A[(i+9)+lda*(k+3)];
                 alpha10 =  A[(i+10)+lda*(k+3)];
                  alpha11 =  A[(i+11)+lda*(k+3)];
                 alpha12 =  A[(i+12)+lda*(k+3)];
                 alpha13 =  A[(i+13)+lda*(k+3)];
                 alpha14 =  A[(i+14)+lda*(k+3)];
                 alpha15 =  A[(i+15)+lda*(k+3)];
                  vc = svmla_m(pg,vc, vb3, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb3, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb3, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb3, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb3, alpha4); // sum += ALPHA*A*B
           	vc5 = svmla_m(pg, vc5, vb3,alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb3, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb3, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb3, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb3, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb3, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb3, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb3, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb3, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb3, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb3, alpha15); // sum += ALPHA*A*B
		}
		flag++;
                }
                for ( int k1 = k; k1 < K; k1 += 1) {
                  //svfloat32_t vb = svld1(pg, &B[((k+(K*itr))*ldb)+0]);
                  svfloat32_t vb = svld1(pg, &B[((k1+(K*(j/ldb)))*ldb)+0]);

                register float alpha =  A[i+lda*k1];
                register float alpha1 =  A[(i+1)+lda*k1];
                register float alpha2 =  A[(i+2)+lda*k1];
                register float alpha3 =  A[(i+3)+lda*k1];
                register float alpha4 =  A[(i+4)+lda*k1];
                register float alpha5 =  A[(i+5)+lda*k1];
                register float alpha6 =  A[(i+6)+lda*k1];
                register float alpha7 =  A[(i+7)+lda*k1];
                register float alpha8 =  A[(i+8)+lda*k1];
                register float alpha9 =  A[(i+9)+lda*k1];
                register float alpha10 =  A[(i+10)+lda*k1];
                 register float alpha11 =  A[(i+11)+lda*k1];
                register float alpha12 =  A[(i+12)+lda*k1];
                register float alpha13 =  A[(i+13)+lda*k1];
                register float alpha14 =  A[(i+14)+lda*k1];
                register float alpha15 =  A[(i+15)+lda*k1];
                  vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
           vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B

        }
                svst1(pg, &C[(i+i1)*ldc+(j+j1)], vc);
                svst1(pg, &C[(i+i1+1)*ldc+(j+j1)], vc1);
                svst1(pg, &C[(i+i1+2)*ldc+(j+j1)], vc2);
                svst1(pg, &C[(i+i1+3)*ldc+(j+j1)], vc3);
                svst1(pg, &C[(i+i1+4)*ldc+(j+j1)], vc4);
                svst1(pg, &C[(i+i1+5)*ldc+(j+j1)], vc5);
                svst1(pg, &C[(i+i1+6)*ldc+(j+j1)], vc6);
                svst1(pg, &C[(i+i1+7)*ldc+(j+j1)], vc7);
                svst1(pg, &C[(i+i1+8)*ldc+(j+j1)], vc8);
                svst1(pg, &C[(i+i1+9)*ldc+(j+j1)], vc9);
                svst1(pg, &C[(i+i1+10)*ldc+(j+j1)], vc10);
                svst1(pg, &C[(i+i1+11)*ldc+(j+j1)], vc11);
                svst1(pg, &C[(i+i1+12)*ldc+(j+j1)], vc12);
                svst1(pg, &C[(i+i1+13)*ldc+(j+j1)], vc13);
                svst1(pg, &C[(i+i1+14)*ldc+(j+j1)], vc14);
                svst1(pg, &C[(i+i1+15)*ldc+(j+j1)], vc15);
        }
	//itr++;
        }

//int k_left = k;
  int i_left=i;
//itr=0; // need to look if necessary
		 #pragma omp for 
  for ( j = 0; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
     svfloat32_t  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     for (i=i_left; i < M; i += 4) {
        vc= svld1(pg, &C[(i+i1)*ldc+(j+j1)]);
        if((i+1) < M)  {vc1= svld1(pg, &C[(i+i1+1)*ldc+(j+j1)]);}
        if ((i+2) < M) {vc2= svld1(pg, &C[(i+i1+2)*ldc+(j+j1)]);}
        if ((i+3) < M) {vc3= svld1(pg, &C[(i+i1+3)*ldc+(j+j1)]);}
       for (int k = 0; k < K; k += 1) {
                alpha =  A[i+lda*k];
                if ((i+1) < M) {alpha1 =  A[(i+1)+lda*k]; }
                if ((i+2) < M) { alpha2 =  A[(i+2)+lda*k];}
                if ((i+3) < M) { alpha3 =  A[(i+3)+lda*k];}
                //vb = svld1(pg, &B[((k+(K*itr))*ldb)+0]);
                vb = svld1(pg, &B[((k+(K*(j/ldb)))*ldb)+0]);
             //   vb = svld1(pg, &B[((k+(K*j)))+0]);
                  vc = svmla_m(pg, vc, vb, alpha); // sum += ALPHA*A*B
                  if ((i+1) < M) {vc1 = svmla_m(pg, vc1, vb, alpha1);} // sum += ALPHA*A*B
                  if ((i+2) < M) {vc2 = svmla_m(pg, vc2, vb, alpha2);} // sum += ALPHA*A*B
                  if ((i+3) < M) {vc3 = svmla_m(pg, vc3, vb, alpha3);}// sum += ALPHA*A*B
        }
          svst1(pg, &C[(i+i1)*ldc+(j+j1)], vc);
          if ((i+1) < M)      {svst1(pg, &C[((i+i1)+1)*ldc+(j+j1)], vc1);}
          if ((i+2) < M)      {svst1(pg, &C[((i+i1)+2)*ldc+(j+j1)], vc2);}
          if ((i+3) < M)      {svst1(pg, &C[((i+i1)+3)*ldc+(j+j1)], vc3);}
     }
//	itr++;
  }
}
}
void gemm_new1( int ii, int jj, int kk, float *A, float *B, float *C, float ALPHA, int M, int N, int K,  int lda,int ldb,int ldc)
{
 #pragma omp parallel
{
		int itr =0;
int i, j, k;
int i1=ii, j1=jj, k1=kk;
//int num = omp_get_num_threads(void);
//printf("%d num of threads ", num);
//M = M/2;
#pragma omp for  
for( j = 0; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
        for (i = 0; i < M-15; i += 16) {
                __builtin_prefetch(B, 0, 2);
                __builtin_prefetch(A, 0, 2);
                __builtin_prefetch(&C[(i+i1)*ldc+(j+j1)], 0, 3);
        svfloat32_t vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, vc16, vc17, vc18, vc19, vc20, vc21, vc22, vc23;// vc24, vc25, vc26, vc27, vc28, vc29, vc30, vc31;
        vc= svld1(pg, &C[(i+i1)*ldc+(j+j1)]);
        vc1= svld1(pg, &C[(i+i1+1)*ldc+(j+j1)]);
        vc2= svld1(pg, &C[(i+i1+2)*ldc+(j+j1)]);
        vc3= svld1(pg, &C[(i+i1+3)*ldc+(j+j1)]);
        vc4= svld1(pg, &C[(i+i1+4)*ldc+(j+j1)]);
        vc5= svld1(pg, &C[(i+i1+5)*ldc+(j+j1)]);
        vc6= svld1(pg, &C[(i+i1+6)*ldc+(j+j1)]);
        vc7= svld1(pg, &C[(i+i1+7)*ldc+(j+j1)]);
        vc8= svld1(pg, &C[(i+i1+8)*ldc+(j+j1)]);
        vc9= svld1(pg, &C[(i+i1+9)*ldc+(j+j1)]);
        vc10= svld1(pg, &C[(i+i1+10)*ldc+(j+j1)]);
        vc11= svld1(pg, &C[(i+i1+11)*ldc+(j+j1)]);
        vc12= svld1(pg, &C[(i+i1+12)*ldc+(j+j1)]);
        vc13= svld1(pg, &C[(i+i1+13)*ldc+(j+j1)]);
        vc14= svld1(pg, &C[(i+i1+14)*ldc+(j+j1)]);
        vc15= svld1(pg, &C[(i+i1+15)*ldc+(j+j1)]);
        svfloat32_t vb,vb1,vb2,vb3,vb4,vb5,vb6,vb7;
                int flag =0;
                __builtin_prefetch(B, 0, 3);
                __builtin_prefetch(A, 0, 3);
		 //#pragma omp for 
        for ( k = 0; k < K; k += 1) {
                  //svfloat32_t vb = svld1(pg, &B[((k+(K*itr))*ldb)+0]);
                  svfloat32_t vb = svld1(pg, &B[((k+(K*(j/ldb)))*ldb)+0]);

                register float alpha =  A[i+lda*k];
                register float alpha1 =  A[(i+1)+lda*k];
                register float alpha2 =  A[(i+2)+lda*k];
                register float alpha3 =  A[(i+3)+lda*k];
                register float alpha4 =  A[(i+4)+lda*k];
                register float alpha5 =  A[(i+5)+lda*k];
                register float alpha6 =  A[(i+6)+lda*k];
                register float alpha7 =  A[(i+7)+lda*k];
                register float alpha8 =  A[(i+8)+lda*k];
                register float alpha9 =  A[(i+9)+lda*k];
                register float alpha10 =  A[(i+10)+lda*k];
                 register float alpha11 =  A[(i+11)+lda*k];
                register float alpha12 =  A[(i+12)+lda*k];
                register float alpha13 =  A[(i+13)+lda*k];
                register float alpha14 =  A[(i+14)+lda*k];
                register float alpha15 =  A[(i+15)+lda*k];
                  vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
           vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B

        }
                svst1(pg, &C[(i+i1)*ldc+(j+j1)], vc);
                svst1(pg, &C[(i+i1+1)*ldc+(j+j1)], vc1);
                svst1(pg, &C[(i+i1+2)*ldc+(j+j1)], vc2);
                svst1(pg, &C[(i+i1+3)*ldc+(j+j1)], vc3);
                svst1(pg, &C[(i+i1+4)*ldc+(j+j1)], vc4);
                svst1(pg, &C[(i+i1+5)*ldc+(j+j1)], vc5);
                svst1(pg, &C[(i+i1+6)*ldc+(j+j1)], vc6);
                svst1(pg, &C[(i+i1+7)*ldc+(j+j1)], vc7);
                svst1(pg, &C[(i+i1+8)*ldc+(j+j1)], vc8);
                svst1(pg, &C[(i+i1+9)*ldc+(j+j1)], vc9);
                svst1(pg, &C[(i+i1+10)*ldc+(j+j1)], vc10);
                svst1(pg, &C[(i+i1+11)*ldc+(j+j1)], vc11);
                svst1(pg, &C[(i+i1+12)*ldc+(j+j1)], vc12);
                svst1(pg, &C[(i+i1+13)*ldc+(j+j1)], vc13);
                svst1(pg, &C[(i+i1+14)*ldc+(j+j1)], vc14);
                svst1(pg, &C[(i+i1+15)*ldc+(j+j1)], vc15);
        }
	//itr++;
        }

//int k_left = k;
  int i_left=i;
//itr=0; // need to look if necessary
		 #pragma omp for  
  for ( j = 0; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
     svfloat32_t  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     for (i=i_left; i < M; i += 4) {
        vc= svld1(pg, &C[(i+i1)*ldc+(j+j1)]);
        if((i+1) < M)  {vc1= svld1(pg, &C[(i+i1+1)*ldc+(j+j1)]);}
        if ((i+2) < M) {vc2= svld1(pg, &C[(i+i1+2)*ldc+(j+j1)]);}
        if ((i+3) < M) {vc3= svld1(pg, &C[(i+i1+3)*ldc+(j+j1)]);}
       for (int k = 0; k < K; k += 1) {
                alpha =  A[i+lda*k];
                if ((i+1) < M) {alpha1 =  A[(i+1)+lda*k]; }
                if ((i+2) < M) { alpha2 =  A[(i+2)+lda*k];}
                if ((i+3) < M) { alpha3 =  A[(i+3)+lda*k];}
                //vb = svld1(pg, &B[((k+(K*itr))*ldb)+0]);
                vb = svld1(pg, &B[((k+(K*(j/ldb)))*ldb)+0]);
             //   vb = svld1(pg, &B[((k+(K*j)))+0]);
                  vc = svmla_m(pg, vc, vb, alpha); // sum += ALPHA*A*B
                  if ((i+1) < M) {vc1 = svmla_m(pg, vc1, vb, alpha1);} // sum += ALPHA*A*B
                  if ((i+2) < M) {vc2 = svmla_m(pg, vc2, vb, alpha2);} // sum += ALPHA*A*B
                  if ((i+3) < M) {vc3 = svmla_m(pg, vc3, vb, alpha3);}// sum += ALPHA*A*B
        }
          svst1(pg, &C[(i+i1)*ldc+(j+j1)], vc);
          if ((i+1) < M)      {svst1(pg, &C[((i+i1)+1)*ldc+(j+j1)], vc1);}
          if ((i+2) < M)      {svst1(pg, &C[((i+i1)+2)*ldc+(j+j1)], vc2);}
          if ((i+3) < M)      {svst1(pg, &C[((i+i1)+3)*ldc+(j+j1)], vc3);}
     }
//	itr++;
  }
}
}







void gemm_new( int ii, int jj, int kk, float *A, float *B, float *C, float ALPHA, int M, int N, int K,  int lda,int ldb,int ldc)
{
//int ii =0, jj=0, kk=0;
// #pragma omp parallel //private(M)
//        {
int i, j, k;
int i1=ii, j1=jj, k1=kk;
//int num = omp_get_num_threads(void);
//printf("%d num of threads ", num);
//M = M/2;
// #pragma omp for 
for( j = 0; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
        for (i = 0; i < M-15; i += 16) {
                __builtin_prefetch(C, 0, 3);
                __builtin_prefetch(B, 0, 2);
                __builtin_prefetch(A, 0, 2);
        svfloat32_t vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, vc16, vc17, vc18, vc19, vc20, vc21, vc22, vc23;// vc24, vc25, vc26, vc27, vc28, vc29, vc30, vc31;
        vc= svld1(pg, &C[(i+i1)*ldc+(j+j1)]);
        vc1= svld1(pg, &C[(i+i1+1)*ldc+(j+j1)]);
        vc2= svld1(pg, &C[(i+i1+2)*ldc+(j+j1)]);
        vc3= svld1(pg, &C[(i+i1+3)*ldc+(j+j1)]);
        vc4= svld1(pg, &C[(i+i1+4)*ldc+(j+j1)]);
        vc5= svld1(pg, &C[(i+i1+5)*ldc+(j+j1)]);
        vc6= svld1(pg, &C[(i+i1+6)*ldc+(j+j1)]);
        vc7= svld1(pg, &C[(i+i1+7)*ldc+(j+j1)]);
        vc8= svld1(pg, &C[(i+i1+8)*ldc+(j+j1)]);
        vc9= svld1(pg, &C[(i+i1+9)*ldc+(j+j1)]);
        vc10= svld1(pg, &C[(i+i1+10)*ldc+(j+j1)]);
        vc11= svld1(pg, &C[(i+i1+11)*ldc+(j+j1)]);
        vc12= svld1(pg, &C[(i+i1+12)*ldc+(j+j1)]);
        vc13= svld1(pg, &C[(i+i1+13)*ldc+(j+j1)]);
        vc14= svld1(pg, &C[(i+i1+14)*ldc+(j+j1)]);
        vc15= svld1(pg, &C[(i+i1+15)*ldc+(j+j1)]);
        svfloat32_t vb,vb1,vb2,vb3,vb4,vb5,vb6,vb7;
                int flag =0;
                __builtin_prefetch(B, 0, 3);
                __builtin_prefetch(A, 0, 3);
        for ( k = 0; k < K; k += 1) {
                  svfloat32_t vb = svld1(pg, &B[k*ldb+j]);

                register float alpha =  A[i+lda*k];
                register float alpha1 =  A[(i+1)+lda*k];
                register float alpha2 =  A[(i+2)+lda*k];
                register float alpha3 =  A[(i+3)+lda*k];
                register float alpha4 =  A[(i+4)+lda*k];
                register float alpha5 =  A[(i+5)+lda*k];
                register float alpha6 =  A[(i+6)+lda*k];
                register float alpha7 =  A[(i+7)+lda*k];
                register float alpha8 =  A[(i+8)+lda*k];
                register float alpha9 =  A[(i+9)+lda*k];
                register float alpha10 =  A[(i+10)+lda*k];
                 register float alpha11 =  A[(i+11)+lda*k];
                register float alpha12 =  A[(i+12)+lda*k];
                register float alpha13 =  A[(i+13)+lda*k];
                register float alpha14 =  A[(i+14)+lda*k];
                register float alpha15 =  A[(i+15)+lda*k];
                  vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
           vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B

        }
                svst1(pg, &C[(i+i1)*ldc+(j+j1)], vc);
                svst1(pg, &C[(i+i1+1)*ldc+(j+j1)], vc1);
                svst1(pg, &C[(i+i1+2)*ldc+(j+j1)], vc2);
                svst1(pg, &C[(i+i1+3)*ldc+(j+j1)], vc3);
                svst1(pg, &C[(i+i1+4)*ldc+(j+j1)], vc4);
                svst1(pg, &C[(i+i1+5)*ldc+(j+j1)], vc5);
                svst1(pg, &C[(i+i1+6)*ldc+(j+j1)], vc6);
                svst1(pg, &C[(i+i1+7)*ldc+(j+j1)], vc7);
                svst1(pg, &C[(i+i1+8)*ldc+(j+j1)], vc8);
                svst1(pg, &C[(i+i1+9)*ldc+(j+j1)], vc9);
                svst1(pg, &C[(i+i1+10)*ldc+(j+j1)], vc10);
                svst1(pg, &C[(i+i1+11)*ldc+(j+j1)], vc11);
                svst1(pg, &C[(i+i1+12)*ldc+(j+j1)], vc12);
                svst1(pg, &C[(i+i1+13)*ldc+(j+j1)], vc13);
                svst1(pg, &C[(i+i1+14)*ldc+(j+j1)], vc14);
                svst1(pg, &C[(i+i1+15)*ldc+(j+j1)], vc15);
        }
        }//}

//int k_left = k;
  int i_left=i;
  for ( j = 0; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
     svfloat32_t  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     for (i=i_left; i < M; i += 4) {
        vc= svld1(pg, &C[(i+i1)*ldc+(j+j1)]);
        if((i+1) < M)  {vc1= svld1(pg, &C[(i+i1+1)*ldc+(j+j1)]);}
        if ((i+2) < M) {vc2= svld1(pg, &C[(i+i1+2)*ldc+(j+j1)]);}
        if ((i+3) < M) {vc3= svld1(pg, &C[(i+i1+3)*ldc+(j+j1)]);}
      for (int k = 0; k < K; k += 1) {
                alpha =  A[i+lda*k];
                if ((i+1) < M) {alpha1 =  A[(i+1)+lda*k]; }
                if ((i+2) < M) { alpha2 =  A[(i+2)+lda*k];}
                if ((i+3) < M) { alpha3 =  A[(i+3)+lda*k];}
                vb = svld1(pg, &B[k*ldb+j]);
                  vc = svmla_m(pg, vc, vb, alpha); // sum += ALPHA*A*B
                  if ((i+1) < M) {vc1 = svmla_m(pg, vc1, vb, alpha1);} // sum += ALPHA*A*B
                  if ((i+2) < M) {vc2 = svmla_m(pg, vc2, vb, alpha2);} // sum += ALPHA*A*B
                  if ((i+3) < M) {vc3 = svmla_m(pg, vc3, vb, alpha3);}// sum += ALPHA*A*B
        }
          svst1(pg, &C[(i+i1)*ldc+(j+j1)], vc);
          if ((i+1) < M)      {svst1(pg, &C[((i+i1)+1)*ldc+(j+j1)], vc1);}
          if ((i+2) < M)      {svst1(pg, &C[((i+i1)+2)*ldc+(j+j1)], vc2);}
          if ((i+3) < M)      {svst1(pg, &C[((i+i1)+3)*ldc+(j+j1)], vc3);}
     }
  }
}

void gemm_nn_pack2(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C,  int ldc, int BlockM, int BlockN, int BlockK, float *transposeB, float *transposeA)

{

//       float *transposeB, *transposeA;
  //      transposeB= (float *)malloc(BlockM*BlockN*BlockK*sizeof(float));
    //    transposeA= (float *)malloc(BlockM*BlockN*BlockK*sizeof(float));
 
	int ld = svcntw();
        int ii,jj,kk,i,j,k;

for (jj = 0; jj < N; jj+=BlockN) {
        int Nc = ((jj+BlockN>N)?(N-jj):(BlockN));
    	for (kk = 0; kk < K; kk+=BlockK) {
        	int Kc = ((kk+BlockK > K)?(K-kk):(BlockK));
                int itr=0;
		#pragma omp parallel for
		for(int j=0;j<Nc;j+=svcntw())
                {
			svbool_t pg = svwhilelt_b32(j, Nc);
			for(int k=0;k<Kc;k++)
                	{
                	//      transposeB[k*Kc+j] = B[(k+kk)*ldb+(j+jj)];
                        	svfloat32_t tmp = svld1(pg, &B[(k+kk)*ldb+(j+jj)]);
                        	//svst1(pg, &transposeB[((k+(Kc*itr))*ld)+0], tmp);
                        	svst1(pg, &transposeB[((k+(Kc*(j/ld)))*ld)+0], tmp);
                        	//transposeB[k*Nc+j] = B[(k+kk)*ldb+(j+jj)];
                	}
			itr++;
                }
        	for (ii = 0; ii < M; ii+=BlockM) {
        		int Mc = ((ii+BlockM >M)?(M-ii):(BlockM)) ;

		#pragma omp parallel for
			for(int k=0;k<Kc;k+=svcntw())
        		{
                		int itr1=0;
                		svbool_t pg = svwhilelt_b32(k, Kc);
                		svuint32_t offset_index = svindex_u32( 0, Mc * sizeof( float ) );
                		for(int i=0;i<Mc;i++)
                		{
                        		svfloat32_t tmp = svld1(pg, &A[(i+ii)*lda+(k+kk)]);
                        		svst1_scatter_offset(pg,&transposeA[k*Mc+i],offset_index, tmp);
                        		//transposeA[k*Mc+i] = A[(i+ii)*lda+(k+kk)];
                		}
        		}
        		//#pragma omp parallel
			//(ii,jj,kk,transposeA,transposeB, C,ALPHA, Mc,Nc, Kc, Mc,ld,ldc );
			gemm_new1(ii,jj,kk,transposeA,transposeB, C,ALPHA, Mc,Nc, Kc, Mc,ld,ldc );
    			//  for (i = 0; i < Block; i++) {
      			//  for (j = 0; j < Block; j++) {
        		//        float tmp = C[(ii+i)*ldc+(jj+j)];
			//          for (k = 0; k < Block; k++) {
	  		//          tmp += A[(ii+i)*lda+(kk+k)]*B[(kk+k)*ldb+(jj+j)];
    			//      }
     	 		//    C[(ii+i)*ldc+(jj+j)] = tmp;
       			// }
     			// }
    		}
  		}
}
	//free(transposeB);
	//transposeB=NULL;
//	free(transposeA);
//	transposeA=NULL;
}
void gemm_nn_pack1(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C,  int ldc, int BlockM, int BlockN, int BlockK)

{
       float *transposeB, *transposeA;
        transposeB= (float *)malloc(BlockM*BlockN*BlockK*sizeof(float));
        transposeA= (float *)malloc(BlockM*BlockN*BlockK*sizeof(float));
        int ii,jj,kk,i,j,k;
 for (jj = 0; jj < N; jj+=BlockN) {
        int Nc = ((jj+BlockN>N)?(N-jj):(BlockN));
    for (kk = 0; kk < K; kk+=BlockK) {
        int Kc = ((kk+BlockK > K)?(K-kk):(BlockK));
                int itr=0;
		for(int j=0;j<Nc;j+=svcntw())
                {
		svbool_t pg = svwhilelt_b32(j, Nc);
                for(int k=0;k<Kc;k++)
                {
                //      transposeB[k*Kc+j] = B[(k+kk)*ldb+(j+jj)];
                        svfloat32_t tmp = svld1(pg, &B[(k+kk)*ldb+(j+jj)]);
                        svst1(pg, &transposeB[((k+(Kc*itr))*16)+0], tmp);
                        //transposeB[k*Nc+j] = B[(k+kk)*ldb+(j+jj)];
                }
		itr++;
                }
        for (ii = 0; ii < M; ii+=BlockM) {
        int Mc = ((ii+BlockM >M)?(M-ii):(BlockM)) ;

for(int i=0;i<Mc;i++)
{
        for(int k=0;k<Kc;k++)
        {
                transposeA[k*Mc+i] = A[(i+ii)*lda+(k+kk)];
        }
}
        gemm_new1(ii,jj,kk,transposeA,transposeB, C,ALPHA, Mc,Nc, Kc, Mc,16,ldc );
    //  for (i = 0; i < Block; i++) {
      //  for (j = 0; j < Block; j++) {
        //        float tmp = C[(ii+i)*ldc+(jj+j)];
//          for (k = 0; k < Block; k++) {
  //          tmp += A[(ii+i)*lda+(kk+k)]*B[(kk+k)*ldb+(jj+j)];
    //      }
      //    C[(ii+i)*ldc+(jj+j)] = tmp;
       // }
     // }
    }
  }
}
free(transposeB);
free(transposeA);
}

void gemm_nn_pack(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C,  int ldc, int BlockM, int BlockN, int BlockK)
{
        //int BlockM = 64, BlockN=1024, BlockK=256;
       // int BlockM = 32, BlockN=2048, BlockK=256;
 //       int BlockM = 128, BlockN=2048, BlockK=256;

       float *transposeB, *transposeA;
        transposeB= (float *)malloc(BlockM*BlockN*BlockK*sizeof(float));
        transposeA= (float *)malloc(BlockM*BlockN*BlockK*sizeof(float));
        int ii,jj,kk,i,j,k;
 for (jj = 0; jj < N; jj+=BlockN) {
        int Nc = ((jj+BlockN>N)?(N-jj):(BlockN));
    for (kk = 0; kk < K; kk+=BlockK) {
        int Kc = ((kk+BlockK > K)?(K-kk):(BlockK));
                for(int k=0;k<Kc;k++)
                {
                for(int j=0;j<Nc;j++)
                {
                       // transposeB[k*Kc+j] = B[(k+kk)*ldb+(j+jj)];
                        transposeB[k*Nc+j] = B[(k+kk)*ldb+(j+jj)];
                }
                }
        for (ii = 0; ii < M; ii+=BlockM) {
        int Mc = ((ii+BlockM >M)?(M-ii):(BlockM)) ;

for(int i=0;i<Mc;i++)
{
        for(int k=0;k<Kc;k++)
        {
                transposeA[k*Mc+i] = A[(i+ii)*lda+(k+kk)];
        }
}
	gemm_new(ii,jj,kk,transposeA,transposeB, C,ALPHA, Mc,Nc, Kc, Mc,Nc,ldc );

    }
  } 
} 
} 

/* multi-core + unroll 16 when alpha !=1*/
void gemm_nn_unroll16(int ii, int jj, int kk, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{

//////// svcntw()  need to store in temp value
 #pragma omp parallel //private(M)
        {
int i=ii, j=jj, k=kk;
 #pragma omp for
  for ( j = 0; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
     for (i = 0; i < M-15; i += 16) {
        svfloat32_t vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15;

        vc= svld1(pg, &C[i*ldc+j]);
        vc1= svld1(pg, &C[(i+1)*ldc+j]);
        vc2= svld1(pg, &C[(i+2)*ldc+j]);
        vc3= svld1(pg, &C[(i+3)*ldc+j]);
        vc4= svld1(pg, &C[(i+4)*ldc+j]);
        vc5= svld1(pg, &C[(i+5)*ldc+j]);
        vc6= svld1(pg, &C[(i+6)*ldc+j]);
        vc7= svld1(pg, &C[(i+7)*ldc+j]);
        vc8= svld1(pg, &C[(i+8)*ldc+j]);
        vc9= svld1(pg, &C[(i+9)*ldc+j]);
        vc10= svld1(pg, &C[(i+10)*ldc+j]);
        vc11= svld1(pg, &C[(i+11)*ldc+j]);
        vc12= svld1(pg, &C[(i+12)*ldc+j]);
        vc13= svld1(pg, &C[(i+13)*ldc+j]);
        vc14= svld1(pg, &C[(i+14)*ldc+j]);
        vc15= svld1(pg, &C[(i+15)*ldc+j]);
        for ( k = 0; k < K; k += 1) {
                svfloat32_t vb = svld1(pg, &B[k*ldb+j]);

                register float alpha = ALPHA * A[i*lda+k];
                register float alpha1 = ALPHA * A[(i+1)*lda+k];
                register float alpha2 = ALPHA * A[(i+2)*lda+k];
                register float alpha3 = ALPHA * A[(i+3)*lda+k];
                register float alpha4 = ALPHA * A[(i+4)*lda+k];
                register float alpha5 = ALPHA * A[(i+5)*lda+k];
                register float alpha6 = ALPHA * A[(i+6)*lda+k];
                register float alpha7 = ALPHA * A[(i+7)*lda+k];
                register float alpha8 = ALPHA * A[(i+8)*lda+k];
                register float alpha9 = ALPHA * A[(i+9)*lda+k];
                register float alpha10 = ALPHA * A[(i+10)*lda+k];
		 register float alpha11 = ALPHA * A[(i+11)*lda+k];
                register float alpha12 = ALPHA * A[(i+12)*lda+k];
                register float alpha13 = ALPHA * A[(i+13)*lda+k];
                register float alpha14 = ALPHA * A[(i+14)*lda+k];
                register float alpha15 = ALPHA * A[(i+15)*lda+k];
                  vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
           vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B
           }
                svst1(pg, &C[i*ldc+j], vc);
                svst1(pg, &C[(i+1)*ldc+j], vc1);
                svst1(pg, &C[(i+2)*ldc+j], vc2);
                svst1(pg, &C[(i+3)*ldc+j], vc3);
                svst1(pg, &C[(i+4)*ldc+j], vc4);
                svst1(pg, &C[(i+5)*ldc+j], vc5);
                svst1(pg, &C[(i+6)*ldc+j], vc6);
                svst1(pg, &C[(i+7)*ldc+j], vc7);
                svst1(pg, &C[(i+8)*ldc+j], vc8);
                svst1(pg, &C[(i+9)*ldc+j], vc9);
                svst1(pg, &C[(i+10)*ldc+j], vc10);
                svst1(pg, &C[(i+11)*ldc+j], vc11);
                svst1(pg, &C[(i+12)*ldc+j], vc12);
                svst1(pg, &C[(i+13)*ldc+j], vc13);
                svst1(pg, &C[(i+14)*ldc+j], vc14);
                svst1(pg, &C[(i+15)*ldc+j], vc15);
        }
     }
int i_left=i;
    #pragma omp for
  for ( j = 0; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
     svfloat32_t  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     for (i=i_left; i < M; i += 4) {
        vc= svld1(pg, &C[i*ldc+j]);
        if((i+1) < M)  {vc1= svld1(pg, &C[(i+1)*ldc+j]);}
        if ((i+2) < M) {vc2= svld1(pg, &C[(i+2)*ldc+j]);}
        if ((i+3) < M) {vc3= svld1(pg, &C[(i+3)*ldc+j]);}
       for (int k = 0; k < K; k += 1) {
                alpha = ALPHA * A[i*lda+k];
                if ((i+1) < M) {alpha1 = ALPHA * A[(i+1)*lda+k]; }
                if ((i+2) < M) { alpha2 = ALPHA * A[(i+2)*lda+k];}
                if ((i+3) < M) { alpha3 = ALPHA * A[(i+3)*lda+k];}
                vb = svld1(pg, &B[k*ldb+j]);
                  vc = svmla_m(pg, vc, vb, alpha); // sum += ALPHA*A*B
                  if ((i+1) < M) {vc1 = svmla_m(pg, vc1, vb, alpha1);} // sum += ALPHA*A*B
                  if ((i+2) < M) {vc2 = svmla_m(pg, vc2, vb, alpha2);} // sum += ALPHA*A*B
                  if ((i+3) < M) {vc3 = svmla_m(pg, vc3, vb, alpha3);}// sum += ALPHA*A*B
        }
          svst1(pg, &C[i*ldc+j], vc);
          if ((i+1) < M)      {svst1(pg, &C[(i+1)*ldc+j], vc1);}
          if ((i+2) < M)      {svst1(pg, &C[(i+2)*ldc+j], vc2);}
          if ((i+3) < M)      {svst1(pg, &C[(i+3)*ldc+j], vc3);}
     }
  }}
}

/* multi-core + unroll 16 when alpha==1*/
void gemm_nn_unroll16_noalpha(int ii, int jj, int kk, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
//#pragma omp parallel //private(M)
        {
int i=ii, j=jj, k=kk;
// #pragma omp for
  for ( j = 0; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
     for (i = 0; i < M-15; i += 16) {
        svfloat32_t vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15;

        vc= svld1(pg, &C[i*ldc+j]);
        vc1= svld1(pg, &C[(i+1)*ldc+j]);
        vc2= svld1(pg, &C[(i+2)*ldc+j]);
        vc3= svld1(pg, &C[(i+3)*ldc+j]);
        vc4= svld1(pg, &C[(i+4)*ldc+j]);
        vc5= svld1(pg, &C[(i+5)*ldc+j]);
        vc6= svld1(pg, &C[(i+6)*ldc+j]);
        vc7= svld1(pg, &C[(i+7)*ldc+j]);
        vc8= svld1(pg, &C[(i+8)*ldc+j]);
        vc9= svld1(pg, &C[(i+9)*ldc+j]);
        vc10= svld1(pg, &C[(i+10)*ldc+j]);
        vc11= svld1(pg, &C[(i+11)*ldc+j]);
        vc12= svld1(pg, &C[(i+12)*ldc+j]);
        vc13= svld1(pg, &C[(i+13)*ldc+j]);
        vc14= svld1(pg, &C[(i+14)*ldc+j]);
        vc15= svld1(pg, &C[(i+15)*ldc+j]);
        for ( k = 0; k < K; k += 1) {
                svfloat32_t vb = svld1(pg, &B[k*ldb+j]);

                register float alpha =  A[i*lda+k];
                register float alpha1 =  A[(i+1)*lda+k];
                register float alpha2 =  A[(i+2)*lda+k];
                register float alpha3 =  A[(i+3)*lda+k];
                register float alpha4 =  A[(i+4)*lda+k];
                register float alpha5 =  A[(i+5)*lda+k];
                register float alpha6 =  A[(i+6)*lda+k];
                register float alpha7 =  A[(i+7)*lda+k];
                register float alpha8 =  A[(i+8)*lda+k];
                register float alpha9 =  A[(i+9)*lda+k];
                register float alpha10 =  A[(i+10)*lda+k];
		 register float alpha11 =  A[(i+11)*lda+k];
                register float alpha12 =  A[(i+12)*lda+k];
                register float alpha13 =  A[(i+13)*lda+k];
                register float alpha14 =  A[(i+14)*lda+k];
                register float alpha15 =  A[(i+15)*lda+k];
                  vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
           vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B
           }
                svst1(pg, &C[i*ldc+j], vc);
                svst1(pg, &C[(i+1)*ldc+j], vc1);
                svst1(pg, &C[(i+2)*ldc+j], vc2);
                svst1(pg, &C[(i+3)*ldc+j], vc3);
                svst1(pg, &C[(i+4)*ldc+j], vc4);
                svst1(pg, &C[(i+5)*ldc+j], vc5);
                svst1(pg, &C[(i+6)*ldc+j], vc6);
                svst1(pg, &C[(i+7)*ldc+j], vc7);
                svst1(pg, &C[(i+8)*ldc+j], vc8);
                svst1(pg, &C[(i+9)*ldc+j], vc9);
                svst1(pg, &C[(i+10)*ldc+j], vc10);
                svst1(pg, &C[(i+11)*ldc+j], vc11);
                svst1(pg, &C[(i+12)*ldc+j], vc12);
                svst1(pg, &C[(i+13)*ldc+j], vc13);
                svst1(pg, &C[(i+14)*ldc+j], vc14);
                svst1(pg, &C[(i+15)*ldc+j], vc15);
        }
     }
int i_left=i;
 //   #pragma omp for
  for ( j = 0; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
     svfloat32_t  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     for (i=i_left; i < M; i += 4) {
        vc= svld1(pg, &C[i*ldc+j]);
        if((i+1) < M)  {vc1= svld1(pg, &C[(i+1)*ldc+j]);}
        if ((i+2) < M) {vc2= svld1(pg, &C[(i+2)*ldc+j]);}
        if ((i+3) < M) {vc3= svld1(pg, &C[(i+3)*ldc+j]);}
       for (int k = 0; k < K; k += 1) {
                alpha =  A[i*lda+k];
                if ((i+1) < M) {alpha1 =  A[(i+1)*lda+k]; }
                if ((i+2) < M) { alpha2 =  A[(i+2)*lda+k];}
                if ((i+3) < M) { alpha3 =  A[(i+3)*lda+k];}
                vb = svld1(pg, &B[k*ldb+j]);
                  vc = svmla_m(pg, vc, vb, alpha); // sum += ALPHA*A*B
                  if ((i+1) < M) {vc1 = svmla_m(pg, vc1, vb, alpha1);} // sum += ALPHA*A*B
                  if ((i+2) < M) {vc2 = svmla_m(pg, vc2, vb, alpha2);} // sum += ALPHA*A*B
                  if ((i+3) < M) {vc3 = svmla_m(pg, vc3, vb, alpha3);}// sum += ALPHA*A*B
        }
          svst1(pg, &C[i*ldc+j], vc);
          if ((i+1) < M)      {svst1(pg, &C[(i+1)*ldc+j], vc1);}
          if ((i+2) < M)      {svst1(pg, &C[(i+2)*ldc+j], vc2);}
          if ((i+3) < M)      {svst1(pg, &C[(i+3)*ldc+j], vc3);}
     }
  }}
}



/* unroll 24 when alpha ==1*/
void gemm_nn1_unroll24(int ii, int jj, int kk, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
 #pragma omp parallel //private(M)
        {
int i=ii, j=jj, k=kk;
 #pragma omp for 
for( j = jj; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
        for (i = ii; i < M-23; i += 24) {
        svfloat32_t vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, vc16, vc17, vc18, vc19, vc20, vc21, vc22, vc23;// vc24, vc25, vc26, vc27, vc28, vc29, vc30, vc31;

        vc= svld1(pg, &C[i*ldc+j]);
        vc1= svld1(pg, &C[(i+1)*ldc+j]);
        vc2= svld1(pg, &C[(i+2)*ldc+j]);
        vc3= svld1(pg, &C[(i+3)*ldc+j]);
        vc4= svld1(pg, &C[(i+4)*ldc+j]);
        vc5= svld1(pg, &C[(i+5)*ldc+j]);
        vc6= svld1(pg, &C[(i+6)*ldc+j]);
        vc7= svld1(pg, &C[(i+7)*ldc+j]);
        vc8= svld1(pg, &C[(i+8)*ldc+j]);
        vc9= svld1(pg, &C[(i+9)*ldc+j]);
        vc10= svld1(pg, &C[(i+10)*ldc+j]);
        vc11= svld1(pg, &C[(i+11)*ldc+j]);
        vc12= svld1(pg, &C[(i+12)*ldc+j]);
        vc13= svld1(pg, &C[(i+13)*ldc+j]);
        vc14= svld1(pg, &C[(i+14)*ldc+j]);
        vc15= svld1(pg, &C[(i+15)*ldc+j]);
        //

        vc16= svld1(pg, &C[(i+16)*ldc+j]);
        vc17= svld1(pg, &C[(i+17)*ldc+j]);
        vc18= svld1(pg, &C[(i+18)*ldc+j]);
        vc19= svld1(pg, &C[(i+19)*ldc+j]);
        vc20= svld1(pg, &C[(i+20)*ldc+j]);
        vc21= svld1(pg, &C[(i+21)*ldc+j]);
        vc22= svld1(pg, &C[(i+22)*ldc+j]);
        vc23= svld1(pg, &C[(i+23)*ldc+j]);
  for ( k = kk; k < K; k += 1) {
                svfloat32_t vb = svld1(pg, &B[k*ldb+j]);
               
               register float alpha =  A[i*lda+k];
               
               register float alpha1 =  A[(i+1)*lda+k];
                
                register float alpha2 = A[(i+2)*lda+k];
                
                register float alpha3 = A[(i+3)*lda+k];
                
                register float alpha4 = A[(i+4)*lda+k];
                
                register float alpha5 =  A[(i+5)*lda+k];
                
                register float alpha6 =  A[(i+6)*lda+k];
                
                register float alpha7 =  A[(i+7)*lda+k];
                
                register float alpha8 =  A[(i+8)*lda+k];
                
                register float alpha9 =  A[(i+9)*lda+k];
                
                register float alpha10 =  A[(i+10)*lda+k];
                
                register float alpha11 =  A[(i+11)*lda+k];
                
                register float alpha12 =  A[(i+12)*lda+k];
                
                register float alpha13 =  A[(i+13)*lda+k];
               
                register float alpha14 =  A[(i+14)*lda+k];
                
                register float alpha15 =  A[(i+15)*lda+k];
                
                register float alpha16 = A[(i+16)*lda+k];
                
                register float alpha17 =  A[(i+17)*lda+k];
                
                register float alpha18 = A[(i+18)*lda+k];
                
                register float alpha19 =  A[(i+19)*lda+k];
                
                register float alpha20 =  A[(i+20)*lda+k];
               
                register float alpha21 =  A[(i+21)*lda+k];
                
                register float alpha22 = A[(i+22)*lda+k];
		
                register float alpha23 = A[(i+23)*lda+k];
              
                 vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
		vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B
                  vc16 = svmla_m(pg, vc16, vb, alpha16); // sum += ALPHA*A*B
                  vc17 = svmla_m(pg, vc17, vb, alpha17); // sum += ALPHA*A*B
                  vc18 = svmla_m(pg, vc18, vb, alpha18); // sum += ALPHA*A*B
                  vc19 = svmla_m(pg, vc19, vb, alpha19); // sum += ALPHA*A*B
                  vc20 = svmla_m(pg, vc20, vb, alpha20); // sum += ALPHA*A*B
                  vc21 = svmla_m(pg, vc21, vb, alpha21); // sum += ALPHA*A*B
                  vc22 = svmla_m(pg, vc22, vb, alpha22); // sum += ALPHA*A*B
                  vc23 = svmla_m(pg, vc23, vb, alpha23); // sum += ALPHA*A*B
                }
                svst1(pg, &C[i*ldc+j], vc);
                svst1(pg, &C[(i+1)*ldc+j], vc1);
                svst1(pg, &C[(i+2)*ldc+j], vc2);
                svst1(pg, &C[(i+3)*ldc+j], vc3);
                svst1(pg, &C[(i+4)*ldc+j], vc4);
                svst1(pg, &C[(i+5)*ldc+j], vc5);
                svst1(pg, &C[(i+6)*ldc+j], vc6);
                svst1(pg, &C[(i+7)*ldc+j], vc7);
                svst1(pg, &C[(i+8)*ldc+j], vc8);
                svst1(pg, &C[(i+9)*ldc+j], vc9);
                svst1(pg, &C[(i+10)*ldc+j], vc10);
                svst1(pg, &C[(i+11)*ldc+j], vc11);
                svst1(pg, &C[(i+12)*ldc+j], vc12);
                svst1(pg, &C[(i+13)*ldc+j], vc13);
                svst1(pg, &C[(i+14)*ldc+j], vc14);
                svst1(pg, &C[(i+15)*ldc+j], vc15);
                svst1(pg, &C[(i+16)*ldc+j], vc16);
                svst1(pg, &C[(i+17)*ldc+j], vc17);
                svst1(pg, &C[(i+18)*ldc+j], vc18);
                svst1(pg, &C[(i+19)*ldc+j], vc19);
                svst1(pg, &C[(i+20)*ldc+j], vc20);
                svst1(pg, &C[(i+21)*ldc+j], vc21);
                svst1(pg, &C[(i+22)*ldc+j], vc22);
                svst1(pg, &C[(i+23)*ldc+j], vc23);
        }
        }
	int i_left=i;
    #pragma omp for
  for ( j = jj; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
     svfloat32_t  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     for (i=i_left; i < M; i += 4) {
        vc= svld1(pg, &C[i*ldc+j]);
        if((i+1) < M)  {vc1= svld1(pg, &C[(i+1)*ldc+j]);}
        if ((i+2) < M) {vc2= svld1(pg, &C[(i+2)*ldc+j]);}
        if ((i+3) < M) {vc3= svld1(pg, &C[(i+3)*ldc+j]);}
       for (int k = kk; k < K; k += 1) {
                alpha =  A[i*lda+k];
                if ((i+1) < M) {alpha1 = A[(i+1)*lda+k]; }
                if ((i+2) < M) { alpha2 = A[(i+2)*lda+k];}
                if ((i+3) < M) { alpha3 = A[(i+3)*lda+k];}
                vb = svld1(pg, &B[k*ldb+j]);
                  vc = svmla_m(pg, vc, vb, alpha); // sum += ALPHA*A*B
                  if ((i+1) < M) {vc1 = svmla_m(pg, vc1, vb, alpha1);} // sum += ALPHA*A*B
                  if ((i+2) < M) {vc2 = svmla_m(pg, vc2, vb, alpha2);} // sum += ALPHA*A*B
                  if ((i+3) < M) {vc3 = svmla_m(pg, vc3, vb, alpha3);}// sum += ALPHA*A*B
        }
          svst1(pg, &C[i*ldc+j], vc);
          if ((i+1) < M)      {svst1(pg, &C[(i+1)*ldc+j], vc1);}
          if ((i+2) < M)      {svst1(pg, &C[(i+2)*ldc+j], vc2);}
          if ((i+3) < M)      {svst1(pg, &C[(i+3)*ldc+j], vc3);}
     }
  }}
}

/* unroll 24  when alpha!=1*/
void gemm_nn1_unroll24_alpha(int ii, int jj, int kk, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
 #pragma omp parallel //private(M)
        {
int i=ii, j=jj, k=kk;
 #pragma omp for 
for( j = jj; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
        for (i = ii; i < M-23; i += 24) {
        svfloat32_t vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, vc16, vc17, vc18, vc19, vc20, vc21, vc22, vc23;// vc24, vc25, vc26, vc27, vc28, vc29, vc30, vc31;

        vc= svld1(pg, &C[i*ldc+j]);
        vc1= svld1(pg, &C[(i+1)*ldc+j]);
        vc2= svld1(pg, &C[(i+2)*ldc+j]);
        vc3= svld1(pg, &C[(i+3)*ldc+j]);
        vc4= svld1(pg, &C[(i+4)*ldc+j]);
        vc5= svld1(pg, &C[(i+5)*ldc+j]);
        vc6= svld1(pg, &C[(i+6)*ldc+j]);
        vc7= svld1(pg, &C[(i+7)*ldc+j]);
        vc8= svld1(pg, &C[(i+8)*ldc+j]);
        vc9= svld1(pg, &C[(i+9)*ldc+j]);
        vc10= svld1(pg, &C[(i+10)*ldc+j]);
        vc11= svld1(pg, &C[(i+11)*ldc+j]);
        vc12= svld1(pg, &C[(i+12)*ldc+j]);
        vc13= svld1(pg, &C[(i+13)*ldc+j]);
        vc14= svld1(pg, &C[(i+14)*ldc+j]);
        vc15= svld1(pg, &C[(i+15)*ldc+j]);
        //

        vc16= svld1(pg, &C[(i+16)*ldc+j]);
        vc17= svld1(pg, &C[(i+17)*ldc+j]);
        vc18= svld1(pg, &C[(i+18)*ldc+j]);
        vc19= svld1(pg, &C[(i+19)*ldc+j]);
        vc20= svld1(pg, &C[(i+20)*ldc+j]);
        vc21= svld1(pg, &C[(i+21)*ldc+j]);
        vc22= svld1(pg, &C[(i+22)*ldc+j]);
        vc23= svld1(pg, &C[(i+23)*ldc+j]);
  for ( k = kk; k < K; k += 1) {
                svfloat32_t vb = svld1(pg, &B[k*ldb+j]);
               
               register float alpha = ALPHA * A[i*lda+k];
               
               register float alpha1 = ALPHA * A[(i+1)*lda+k];
          
                register float alpha2 = ALPHA * A[(i+2)*lda+k];
                
                register float alpha3 = ALPHA * A[(i+3)*lda+k];
                
                register float alpha4 = ALPHA * A[(i+4)*lda+k];
                
                register float alpha5 = ALPHA * A[(i+5)*lda+k];
                
                register float alpha6 = ALPHA * A[(i+6)*lda+k];
                
                register float alpha7 = ALPHA * A[(i+7)*lda+k];
                
                register float alpha8 = ALPHA * A[(i+8)*lda+k];
                
                register float alpha9 = ALPHA * A[(i+9)*lda+k];
                
                register float alpha10 = ALPHA * A[(i+10)*lda+k];
               
                register float alpha11 = ALPHA * A[(i+11)*lda+k];
                
                register float alpha12 = ALPHA * A[(i+12)*lda+k];
               
                register float alpha13 = ALPHA * A[(i+13)*lda+k];
               
                register float alpha14 = ALPHA * A[(i+14)*lda+k];
                
                register float alpha15 = ALPHA * A[(i+15)*lda+k];
                
                register float alpha16 = ALPHA * A[(i+16)*lda+k];
               
                register float alpha17 = ALPHA * A[(i+17)*lda+k];
                
                register float alpha18 = ALPHA * A[(i+18)*lda+k];
               
                register float alpha19 = ALPHA * A[(i+19)*lda+k];
                
                register float alpha20 = ALPHA * A[(i+20)*lda+k];
              
                register float alpha21 = ALPHA * A[(i+21)*lda+k];
                
                register float alpha22 = ALPHA * A[(i+22)*lda+k];
                
	        register float alpha23 = ALPHA * A[(i+23)*lda+k];
                
             
                 vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
		vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B
                  vc16 = svmla_m(pg, vc16, vb, alpha16); // sum += ALPHA*A*B
                  vc17 = svmla_m(pg, vc17, vb, alpha17); // sum += ALPHA*A*B
                  vc18 = svmla_m(pg, vc18, vb, alpha18); // sum += ALPHA*A*B
                  vc19 = svmla_m(pg, vc19, vb, alpha19); // sum += ALPHA*A*B
                  vc20 = svmla_m(pg, vc20, vb, alpha20); // sum += ALPHA*A*B
                  vc21 = svmla_m(pg, vc21, vb, alpha21); // sum += ALPHA*A*B
                  vc22 = svmla_m(pg, vc22, vb, alpha22); // sum += ALPHA*A*B
                  vc23 = svmla_m(pg, vc23, vb, alpha23); // sum += ALPHA*A*B
                }
                svst1(pg, &C[i*ldc+j], vc);
                svst1(pg, &C[(i+1)*ldc+j], vc1);
                svst1(pg, &C[(i+2)*ldc+j], vc2);
                svst1(pg, &C[(i+3)*ldc+j], vc3);
                svst1(pg, &C[(i+4)*ldc+j], vc4);
                svst1(pg, &C[(i+5)*ldc+j], vc5);
                svst1(pg, &C[(i+6)*ldc+j], vc6);
                svst1(pg, &C[(i+7)*ldc+j], vc7);
                svst1(pg, &C[(i+8)*ldc+j], vc8);
                svst1(pg, &C[(i+9)*ldc+j], vc9);
                svst1(pg, &C[(i+10)*ldc+j], vc10);
                svst1(pg, &C[(i+11)*ldc+j], vc11);
                svst1(pg, &C[(i+12)*ldc+j], vc12);
                svst1(pg, &C[(i+13)*ldc+j], vc13);
                svst1(pg, &C[(i+14)*ldc+j], vc14);
                svst1(pg, &C[(i+15)*ldc+j], vc15);
                svst1(pg, &C[(i+16)*ldc+j], vc16);
                svst1(pg, &C[(i+17)*ldc+j], vc17);
                svst1(pg, &C[(i+18)*ldc+j], vc18);
                svst1(pg, &C[(i+19)*ldc+j], vc19);
                svst1(pg, &C[(i+20)*ldc+j], vc20);
                svst1(pg, &C[(i+21)*ldc+j], vc21);
                svst1(pg, &C[(i+22)*ldc+j], vc22);
                svst1(pg, &C[(i+23)*ldc+j], vc23);
        }
        }//}
	int i_left=i;
    #pragma omp for
  for ( j = jj; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
     svfloat32_t  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     for (i=i_left; i < M; i += 4) {
        vc= svld1(pg, &C[i*ldc+j]);
        if((i+1) < M)  {vc1= svld1(pg, &C[(i+1)*ldc+j]);}
        if ((i+2) < M) {vc2= svld1(pg, &C[(i+2)*ldc+j]);}
        if ((i+3) < M) {vc3= svld1(pg, &C[(i+3)*ldc+j]);}
       for (int k = kk; k < K; k += 1) {
                alpha = ALPHA * A[i*lda+k];
                if ((i+1) < M) {alpha1 = ALPHA * A[(i+1)*lda+k]; }
                if ((i+2) < M) { alpha2 = ALPHA * A[(i+2)*lda+k];}
                if ((i+3) < M) { alpha3 = ALPHA * A[(i+3)*lda+k];}
                vb = svld1(pg, &B[k*ldb+j]);
                  vc = svmla_m(pg, vc, vb, alpha); // sum += ALPHA*A*B
                  if ((i+1) < M) {vc1 = svmla_m(pg, vc1, vb, alpha1);} // sum += ALPHA*A*B
                  if ((i+2) < M) {vc2 = svmla_m(pg, vc2, vb, alpha2);} // sum += ALPHA*A*B
                  if ((i+3) < M) {vc3 = svmla_m(pg, vc3, vb, alpha3);}// sum += ALPHA*A*B
        }
          svst1(pg, &C[i*ldc+j], vc);
          if ((i+1) < M)      {svst1(pg, &C[(i+1)*ldc+j], vc1);}
          if ((i+2) < M)      {svst1(pg, &C[(i+2)*ldc+j], vc2);}
          if ((i+3) < M)      {svst1(pg, &C[(i+3)*ldc+j], vc3);}
     }
  }}
}



/* unroll 16 + K4 + double buffer when apha==1*/

void gemm_nn1(int ii, int jj, int kk, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
int vl = svcntw();
 #pragma omp parallel //private(M)
        {
int i=ii, j=jj, k=kk;
//int num = omp_get_num_threads(void);
//printf("%d num of threads ", num);
//M = M/2;
 #pragma omp for 
for( j = jj; j < N; j+= vl) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
        for (i = ii; i < M-15; i += 16) {
        svfloat32_t vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, vc16, vc17, vc18, vc19, vc20, vc21, vc22, vc23;// vc24, vc25, vc26, vc27, vc28, vc29, vc30, vc31;

        vc= svld1(pg, &C[i*ldc+j]);
        vc1= svld1(pg, &C[(i+1)*ldc+j]);
        vc2= svld1(pg, &C[(i+2)*ldc+j]);
        vc3= svld1(pg, &C[(i+3)*ldc+j]);
        vc4= svld1(pg, &C[(i+4)*ldc+j]);
        vc5= svld1(pg, &C[(i+5)*ldc+j]);
        vc6= svld1(pg, &C[(i+6)*ldc+j]);
        vc7= svld1(pg, &C[(i+7)*ldc+j]);
        vc8= svld1(pg, &C[(i+8)*ldc+j]);
        vc9= svld1(pg, &C[(i+9)*ldc+j]);
        vc10= svld1(pg, &C[(i+10)*ldc+j]);
        vc11= svld1(pg, &C[(i+11)*ldc+j]);
        vc12= svld1(pg, &C[(i+12)*ldc+j]);
        vc13= svld1(pg, &C[(i+13)*ldc+j]);
        vc14= svld1(pg, &C[(i+14)*ldc+j]);
        vc15= svld1(pg, &C[(i+15)*ldc+j]);
	
    	svfloat32_t vb,vb1,vb2,vb3,vb4,vb5,vb6,vb7;
	int flag =0;
        for ( k = kk; k < K-3; k += 4) {
		if (flag==0){
		 vb = svld1(pg, &B[k*ldb+j]);
		vb1 = svld1(pg, &B[(k+1)*ldb+j]);
		vb2 = svld1(pg, &B[(k+2)*ldb+j]);
		vb3 = svld1(pg, &B[(k+3)*ldb+j]);


		 vb4 = svld1(pg, &B[(k+4)*ldb+j]);
                vb5 = svld1(pg, &B[(k+5)*ldb+j]);
                vb6 = svld1(pg, &B[(k+6)*ldb+j]);
                vb7 = svld1(pg, &B[(k+7)*ldb+j]);


		}
		else
		{
			if(flag & 1)	//odd number
			{
				if(k<K-4)
				{
                                        vb = svld1(pg, &B[(k+4)*ldb+j]);
                                        vb1 = svld1(pg, &B[(k+5)*ldb+j]);
                                        vb2 = svld1(pg, &B[(k+6)*ldb+j]);
                                        vb3 = svld1(pg, &B[(k+7)*ldb+j]);
				}
			}			
			else  //even number
			{
				if(k<K-4)
                                {
					vb4 = svld1(pg, &B[(k+4)*ldb+j]);
                			vb5 = svld1(pg, &B[(k+5)*ldb+j]);
                			vb6 = svld1(pg, &B[(k+6)*ldb+j]);
                			vb7 = svld1(pg, &B[(k+7)*ldb+j]);
                                }
			}
		}
		if(flag & 1)
		{

	       	
	       	register float alpha = A[i*lda+k];
	       	register float alpha0 = A[i*lda+(k+1)];
               	
               	register float alpha1 =  A[(i+1)*lda+k];
               register float alpha01 =  A[(i+1)*lda+(k+1)];
                
                register float alpha2 =  A[(i+2)*lda+k];
                register float alpha02 =  A[(i+2)*lda+(k+1)];
                
                register float alpha3 =  A[(i+3)*lda+k];
                register float alpha03 =  A[(i+3)*lda+(k+1)];
                
                register float alpha4 =  A[(i+4)*lda+k];
                register float alpha04 =  A[(i+4)*lda+(k+1)];
                
                register float alpha5 = A[(i+5)*lda+k];
                register float alpha05 = A[(i+5)*lda+(k+1)];
 		
 		register float alpha6 = A[(i+6)*lda+k];
 		register float alpha06 = A[(i+6)*lda+(k+1)];
                
                register float alpha7 =  A[(i+7)*lda+k];
                register float alpha07 =  A[(i+7)*lda+(k+1)];
               
                register float alpha8 =  A[(i+8)*lda+k];
                register float alpha08 =  A[(i+8)*lda+(k+1)];
                
                register float alpha9 = A[(i+9)*lda+k];
                register float alpha09 = A[(i+9)*lda+(k+1)];
               
                register float alpha10 =  A[(i+10)*lda+k];
                register float alpha010 =  A[(i+10)*lda+(k+1)];
                
                register float alpha11 =  A[(i+11)*lda+k];
                register float alpha011 =  A[(i+11)*lda+(k+1)];
                
                register float alpha12 =  A[(i+12)*lda+k];
                register float alpha012 =  A[(i+12)*lda+(k+1)];
               
                register float alpha13 =  A[(i+13)*lda+k];
                register float alpha013 =  A[(i+13)*lda+(k+1)];
                
                register float alpha14 = A[(i+14)*lda+k];
                register float alpha014 = A[(i+14)*lda+(k+1)];
                
                register float alpha15 =  A[(i+15)*lda+k];
                register float alpha015 =  A[(i+15)*lda+(k+1)];

		vc = svmla_m(pg,vc, vb4, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb5, alpha0); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb4, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb5, alpha01); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb4, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb5, alpha02); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb4, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb5, alpha03); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb4, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb5, alpha04); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb4, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb5, alpha05); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb4, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb5, alpha06); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb4, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb5, alpha07); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb4, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb5, alpha08); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb4, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb5, alpha09); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb4, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb5, alpha010); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb4, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb5, alpha011); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb4, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb5, alpha012); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb4, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb5, alpha013); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb4, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb5, alpha014); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb4, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb5, alpha015); // sum += ALPHA*A*B
	
		alpha = A[i*lda+(k+2)];
	       	alpha0 = A[i*lda+(k+3)];
               
		 alpha1 =  A[(i+1)*lda+(k+2)];
                alpha01 =  A[(i+1)*lda+(k+3)];
                
		 alpha2 =  A[(i+2)*lda+(k+2)];
                alpha02 =  A[(i+2)*lda+(k+3)];
                
		alpha3 =  A[(i+3)*lda+(k+2)];
                alpha03 =  A[(i+3)*lda+(k+3)];
                
		alpha4 =  A[(i+4)*lda+(k+2)];
                alpha04 =  A[(i+4)*lda+(k+3)];
               
                alpha5 = A[(i+5)*lda+(k+2)];
                alpha05 = A[(i+5)*lda+(k+3)];
 		
 		 alpha6 = A[(i+6)*lda+(k+2)];
 		 alpha06 = A[(i+6)*lda+(k+3)];
                
                 alpha7 =  A[(i+7)*lda+(k+2)];
                alpha07 =  A[(i+7)*lda+(k+3)];
                
                alpha8 =  A[(i+8)*lda+(k+2)];
                 alpha08 =  A[(i+8)*lda+(k+3)];
                
                alpha9 = A[(i+9)*lda+(k+2)];
                alpha09 = A[(i+9)*lda+(k+3)];
                
                alpha10 =  A[(i+10)*lda+(k+2)];
                alpha010 =  A[(i+10)*lda+(k+3)];
                
                alpha11 =  A[(i+11)*lda+(k+2)];
                alpha011 =  A[(i+11)*lda+(k+3)];
                
                 alpha12 =  A[(i+12)*lda+(k+2)];
                 alpha012 =  A[(i+12)*lda+(k+3)];
                
                 alpha13 =  A[(i+13)*lda+(k+2)];
                alpha013 =  A[(i+13)*lda+(k+3)];
               
                alpha14 = A[(i+14)*lda+(k+2)];
		alpha014 = A[(i+14)*lda+(k+3)];
                
                alpha15 =  A[(i+15)*lda+(k+2)];
                 alpha015 =  A[(i+15)*lda+(k+3)];
                 vc = svmla_m(pg,vc, vb6, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb7, alpha0); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb6, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb7, alpha01); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb6, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb7, alpha02); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb6, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb7, alpha03); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb6, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb7, alpha04); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb6, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb7, alpha05); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb6, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb7, alpha06); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb6, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb7, alpha07); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb6, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb7, alpha08); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb6, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb7, alpha09); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb6, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb7, alpha010); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb6, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb7, alpha011); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb6, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb7, alpha012); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb6, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb7, alpha013); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb6, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb7, alpha014); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb6, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb7, alpha015); // sum += ALPHA*A*B
			
		}
		else
		{
		

	      
	       register float alpha = A[i*lda+k];
	       register float alpha0 = A[i*lda+(k+1)];
               
               register float alpha1 =  A[(i+1)*lda+k];
               register float alpha01 =  A[(i+1)*lda+(k+1)];
                
                register float alpha2 =  A[(i+2)*lda+k];
                register float alpha02 =  A[(i+2)*lda+(k+1)];
                
                register float alpha3 =  A[(i+3)*lda+k];
                register float alpha03 =  A[(i+3)*lda+(k+1)];
              
                register float alpha4 =  A[(i+4)*lda+k];
                register float alpha04 =  A[(i+4)*lda+(k+1)];
                
                register float alpha5 = A[(i+5)*lda+k];
                register float alpha05 = A[(i+5)*lda+(k+1)];
 		
 		register float alpha6 = A[(i+6)*lda+k];
 		register float alpha06 = A[(i+6)*lda+(k+1)];
               
                register float alpha7 =  A[(i+7)*lda+k];
                register float alpha07 =  A[(i+7)*lda+(k+1)];
                
                register float alpha8 =  A[(i+8)*lda+k];
                register float alpha08 =  A[(i+8)*lda+(k+1)];
               
                register float alpha9 = A[(i+9)*lda+k];
                register float alpha09 = A[(i+9)*lda+(k+1)];
               
                register float alpha10 =  A[(i+10)*lda+k];
                register float alpha010 =  A[(i+10)*lda+(k+1)];
                
                register float alpha11 =  A[(i+11)*lda+k];
                register float alpha011 =  A[(i+11)*lda+(k+1)];
               
                register float alpha12 =  A[(i+12)*lda+k];
                register float alpha012 =  A[(i+12)*lda+(k+1)];
                
                register float alpha13 =  A[(i+13)*lda+k];
                register float alpha013 =  A[(i+13)*lda+(k+1)];
               
                register float alpha14 = A[(i+14)*lda+k];
                register float alpha014 = A[(i+14)*lda+(k+1)];
                
                register float alpha15 =  A[(i+15)*lda+k];
                register float alpha015 =  A[(i+15)*lda+(k+1)];

		/*for unroll4 uncomment below*/ 
		vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb1, alpha0); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb1, alpha01); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb1, alpha02); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb1, alpha03); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb1, alpha04); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb1, alpha05); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb1, alpha06); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb1, alpha07); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb1, alpha08); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb1, alpha09); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb1, alpha010); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb1, alpha011); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb1, alpha012); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb1, alpha013); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb1, alpha014); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb1, alpha015); // sum += ALPHA*A*B
	
		alpha = A[i*lda+(k+2)];
	       	alpha0 = A[i*lda+(k+3)];

                alpha1 =  A[(i+1)*lda+(k+2)];
                alpha01 =  A[(i+1)*lda+(k+3)];

                 alpha2 =  A[(i+2)*lda+(k+2)];
                alpha02 =  A[(i+2)*lda+(k+3)];

                alpha3 =  A[(i+3)*lda+(k+2)];
                alpha03 =  A[(i+3)*lda+(k+3)];

                alpha4 =  A[(i+4)*lda+(k+2)];
                alpha04 =  A[(i+4)*lda+(k+3)];
                
                alpha5 = A[(i+5)*lda+(k+2)];
                alpha05 = A[(i+5)*lda+(k+3)];
 		
 		 alpha6 = A[(i+6)*lda+(k+2)];
 		 alpha06 = A[(i+6)*lda+(k+3)];
                
                 alpha7 =  A[(i+7)*lda+(k+2)];
                alpha07 =  A[(i+7)*lda+(k+3)];
                
                alpha8 =  A[(i+8)*lda+(k+2)];
                 alpha08 =  A[(i+8)*lda+(k+3)];
               
                alpha9 = A[(i+9)*lda+(k+2)];
                alpha09 = A[(i+9)*lda+(k+3)];
                
                alpha10 =  A[(i+10)*lda+(k+2)];
                alpha010 =  A[(i+10)*lda+(k+3)];
                
                alpha11 =  A[(i+11)*lda+(k+2)];
                alpha011 =  A[(i+11)*lda+(k+3)];
                
                 alpha12 =  A[(i+12)*lda+(k+2)];
                 alpha012 =  A[(i+12)*lda+(k+3)];
                
                 alpha13 =  A[(i+13)*lda+(k+2)];
                alpha013 =  A[(i+13)*lda+(k+3)];
               
                alpha14 = A[(i+14)*lda+(k+2)];
		alpha014 = A[(i+14)*lda+(k+3)];
                
                alpha15 =  A[(i+15)*lda+(k+2)];
                 alpha015 =  A[(i+15)*lda+(k+3)];
                 vc = svmla_m(pg,vc, vb2, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb3, alpha0); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb2, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb3, alpha01); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb2, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb3, alpha02); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb2, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb3, alpha03); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb2, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb3, alpha04); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb2, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb3, alpha05); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb2, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb3, alpha06); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb2, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb3, alpha07); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb2, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb3, alpha08); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb2, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb3, alpha09); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb2, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb3, alpha010); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb2, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb3, alpha011); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb2, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb3, alpha012); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb2, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb3, alpha013); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb2, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb3, alpha014); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb2, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb3, alpha015); // sum += ALPHA*A*B
		
		}		
		flag++;
		}
		for ( int k1 = k; k1 < K; k1 += 1) {
                svfloat32_t vb = svld1(pg, &B[k1*ldb+j]);

                register float alpha = ALPHA * A[i*lda+k1];
                register float alpha1 = ALPHA * A[(i+1)*lda+k1];
                register float alpha2 = ALPHA * A[(i+2)*lda+k1];
                register float alpha3 = ALPHA * A[(i+3)*lda+k1];
                register float alpha4 = ALPHA * A[(i+4)*lda+k1];
                register float alpha5 = ALPHA * A[(i+5)*lda+k1];
                register float alpha6 = ALPHA * A[(i+6)*lda+k1];
                register float alpha7 = ALPHA * A[(i+7)*lda+k1];
                register float alpha8 = ALPHA * A[(i+8)*lda+k1];
                register float alpha9 = ALPHA * A[(i+9)*lda+k1];
                register float alpha10 = ALPHA * A[(i+10)*lda+k1];
                register float alpha11 = ALPHA * A[(i+11)*lda+k1];
                register float alpha12 = ALPHA * A[(i+12)*lda+k1];
                register float alpha13 = ALPHA * A[(i+13)*lda+k1];
                register float alpha14 = ALPHA * A[(i+14)*lda+k1];
                register float alpha15 = ALPHA * A[(i+15)*lda+k1];
                 vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B
        }
                svst1(pg, &C[i*ldc+j], vc);
                svst1(pg, &C[(i+1)*ldc+j], vc1);
                svst1(pg, &C[(i+2)*ldc+j], vc2);
                svst1(pg, &C[(i+3)*ldc+j], vc3);
                svst1(pg, &C[(i+4)*ldc+j], vc4);
                svst1(pg, &C[(i+5)*ldc+j], vc5);
                svst1(pg, &C[(i+6)*ldc+j], vc6);
                svst1(pg, &C[(i+7)*ldc+j], vc7);
                svst1(pg, &C[(i+8)*ldc+j], vc8);
                svst1(pg, &C[(i+9)*ldc+j], vc9);
                svst1(pg, &C[(i+10)*ldc+j], vc10);
                svst1(pg, &C[(i+11)*ldc+j], vc11);
                svst1(pg, &C[(i+12)*ldc+j], vc12);
                svst1(pg, &C[(i+13)*ldc+j], vc13);
                svst1(pg, &C[(i+14)*ldc+j], vc14);
                svst1(pg, &C[(i+15)*ldc+j], vc15);
        }
        }//}

//int k_left = k;
  int i_left=i;
    #pragma omp for
  for ( j = jj; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
     svfloat32_t  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     for (i=i_left; i < M; i += 4) {
        vc= svld1(pg, &C[i*ldc+j]);
        if((i+1) < M)  {vc1= svld1(pg, &C[(i+1)*ldc+j]);}
        if ((i+2) < M) {vc2= svld1(pg, &C[(i+2)*ldc+j]);}
        if ((i+3) < M) {vc3= svld1(pg, &C[(i+3)*ldc+j]);}
       for (int k = kk; k < K; k += 1) {
                alpha =  A[i*lda+k];
                if ((i+1) < M) {alpha1 =  A[(i+1)*lda+k]; }
                if ((i+2) < M) { alpha2 =  A[(i+2)*lda+k];}
                if ((i+3) < M) { alpha3 =  A[(i+3)*lda+k];}
                vb = svld1(pg, &B[k*ldb+j]);
                  vc = svmla_m(pg, vc, vb, alpha); // sum += ALPHA*A*B
                  if ((i+1) < M) {vc1 = svmla_m(pg, vc1, vb, alpha1);} // sum += ALPHA*A*B
                  if ((i+2) < M) {vc2 = svmla_m(pg, vc2, vb, alpha2);} // sum += ALPHA*A*B
                  if ((i+3) < M) {vc3 = svmla_m(pg, vc3, vb, alpha3);}// sum += ALPHA*A*B
        }
          svst1(pg, &C[i*ldc+j], vc);
          if ((i+1) < M)      {svst1(pg, &C[(i+1)*ldc+j], vc1);}
          if ((i+2) < M)      {svst1(pg, &C[(i+2)*ldc+j], vc2);}
          if ((i+3) < M)      {svst1(pg, &C[(i+3)*ldc+j], vc3);}
     }
  }}
}


/***************** unroll 16 K 8 + double buffer when alpha==1 ***************/

void gemm_nn_unroll16k8_doublebuffer(int ii, int jj, int kk, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
 #pragma omp parallel //private(M)
        {
int i=ii, j=jj, k=kk;

 #pragma omp for 
for( j = jj; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
        for (i = ii; i < M-15; i += 16) {
        svfloat32_t vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, vc16, vc17, vc18, vc19, vc20, vc21, vc22, vc23;// vc24, vc25, vc26, vc27, vc28, vc29, vc30, vc31;

        vc= svld1(pg, &C[i*ldc+j]);
        vc1= svld1(pg, &C[(i+1)*ldc+j]);
        vc2= svld1(pg, &C[(i+2)*ldc+j]);
        vc3= svld1(pg, &C[(i+3)*ldc+j]);
        vc4= svld1(pg, &C[(i+4)*ldc+j]);
        vc5= svld1(pg, &C[(i+5)*ldc+j]);
        vc6= svld1(pg, &C[(i+6)*ldc+j]);
        vc7= svld1(pg, &C[(i+7)*ldc+j]);
        vc8= svld1(pg, &C[(i+8)*ldc+j]);
        vc9= svld1(pg, &C[(i+9)*ldc+j]);
        vc10= svld1(pg, &C[(i+10)*ldc+j]);
        vc11= svld1(pg, &C[(i+11)*ldc+j]);
        vc12= svld1(pg, &C[(i+12)*ldc+j]);
        vc13= svld1(pg, &C[(i+13)*ldc+j]);
        vc14= svld1(pg, &C[(i+14)*ldc+j]);
        vc15= svld1(pg, &C[(i+15)*ldc+j]);
	
    	svfloat32_t vb,vb1,vb2,vb3,vb4,vb5,vb6,vb7, vb8, vb9, vb10, vb11, vb12, vb13, vb14, vb15;
	int flag =0;
	if(K>15){
	 for ( k = kk; k < K-15; k += 8) {
		if (flag==0){
		 vb = svld1(pg, &B[k*ldb+j]);
		vb1 = svld1(pg, &B[(k+1)*ldb+j]);
		vb2 = svld1(pg, &B[(k+2)*ldb+j]);
		vb3 = svld1(pg, &B[(k+3)*ldb+j]);
		 vb4 = svld1(pg, &B[(k+4)*ldb+j]);
                vb5 = svld1(pg, &B[(k+5)*ldb+j]);
                vb6 = svld1(pg, &B[(k+6)*ldb+j]);
                vb7 = svld1(pg, &B[(k+7)*ldb+j]);
		vb8 = svld1(pg, &B[(k+8)*ldb+j]);
		vb9 = svld1(pg, &B[(k+9)*ldb+j]);
		vb10 = svld1(pg, &B[(k+10)*ldb+j]);
		vb11 = svld1(pg, &B[(k+11)*ldb+j]);
		 vb12 = svld1(pg, &B[(k+12)*ldb+j]);
                vb13 = svld1(pg, &B[(k+13)*ldb+j]);
                vb14 = svld1(pg, &B[(k+14)*ldb+j]);
                vb15 = svld1(pg, &B[(k+15)*ldb+j]);
//


		}
		else
		{
			if(flag & 1)	//odd number
			{
		
				if(k<K-8)
				{
                                        vb = svld1(pg, &B[(k+8)*ldb+j]);
                                        vb1 = svld1(pg, &B[(k+9)*ldb+j]);
                                        vb2 = svld1(pg, &B[(k+10)*ldb+j]);
                                        vb3 = svld1(pg, &B[(k+11)*ldb+j]);
				      vb4 = svld1(pg, &B[(k+12)*ldb+j]);
                                        vb5 = svld1(pg, &B[(k+13)*ldb+j]);
                                        vb6 = svld1(pg, &B[(k+14)*ldb+j]);
                                        vb7 = svld1(pg, &B[(k+15)*ldb+j]);
				}
			}			
			else  //even number
			{
				if(k<K-8)
                                {
				vb8 = svld1(pg, &B[(k+8)*ldb+j]);
                			vb9 = svld1(pg, &B[(k+9)*ldb+j]);
                			vb10 = svld1(pg, &B[(k+10)*ldb+j]);
                			vb11 = svld1(pg, &B[(k+11)*ldb+j]);
				vb12 = svld1(pg, &B[(k+12)*ldb+j]);
                			vb13 = svld1(pg, &B[(k+13)*ldb+j]);
                			vb14 = svld1(pg, &B[(k+14)*ldb+j]);
                			vb15 = svld1(pg, &B[(k+15)*ldb+j]);
                                }
			}
		}
		if(flag & 1)
		{
//		printf(" \nHi I am %d even ", flag);
	       	register float alpha = A[i*lda+k];
	       	register float alpha0 = A[i*lda+(k+1)];

               	register float alpha1 =  A[(i+1)*lda+k];
               register float alpha01 =  A[(i+1)*lda+(k+1)];
                
                register float alpha2 =  A[(i+2)*lda+k];
                register float alpha02 =  A[(i+2)*lda+(k+1)];
                
                register float alpha3 =  A[(i+3)*lda+k];
                register float alpha03 =  A[(i+3)*lda+(k+1)];
                
                register float alpha4 =  A[(i+4)*lda+k];
                register float alpha04 =  A[(i+4)*lda+(k+1)];

                register float alpha5 = A[(i+5)*lda+k];
                register float alpha05 = A[(i+5)*lda+(k+1)];
 		
 		register float alpha6 = A[(i+6)*lda+k];
 		register float alpha06 = A[(i+6)*lda+(k+1)];
               
                register float alpha7 =  A[(i+7)*lda+k];
                register float alpha07 =  A[(i+7)*lda+(k+1)];
                
                register float alpha8 =  A[(i+8)*lda+k];
                register float alpha08 =  A[(i+8)*lda+(k+1)];
                
                register float alpha9 = A[(i+9)*lda+k];
                register float alpha09 = A[(i+9)*lda+(k+1)];
                
                register float alpha10 =  A[(i+10)*lda+k];
                register float alpha010 =  A[(i+10)*lda+(k+1)];
                
                register float alpha11 =  A[(i+11)*lda+k];
                register float alpha011 =  A[(i+11)*lda+(k+1)];
                
                register float alpha12 =  A[(i+12)*lda+k];
                register float alpha012 =  A[(i+12)*lda+(k+1)];
                
                register float alpha13 =  A[(i+13)*lda+k];
                register float alpha013 =  A[(i+13)*lda+(k+1)];
                
                register float alpha14 = A[(i+14)*lda+k];
                register float alpha014 = A[(i+14)*lda+(k+1)];
                
                register float alpha15 =  A[(i+15)*lda+k];
                register float alpha015 =  A[(i+15)*lda+(k+1)];

		vc = svmla_m(pg,vc, vb8, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb9, alpha0); // sum += ALPHA*A*B

                  vc1 = svmla_m(pg, vc1, vb8, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb9, alpha01); // sum += ALPHA*A*B

                  vc2 = svmla_m(pg, vc2, vb8, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb9, alpha02); // sum += ALPHA*A*B

                  vc3 = svmla_m(pg, vc3, vb8, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb9, alpha03); // sum += ALPHA*A*B

                  vc4 = svmla_m(pg, vc4, vb8, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb9, alpha04); // sum += ALPHA*A*B

                  vc5 = svmla_m(pg, vc5, vb8, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb9, alpha05); // sum += ALPHA*A*B

                  vc6= svmla_m(pg, vc6, vb8, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb9, alpha06); // sum += ALPHA*A*B

                  vc7 = svmla_m(pg, vc7, vb8, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb9, alpha07); // sum += ALPHA*A*B

                  vc8 = svmla_m(pg, vc8, vb8, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb9, alpha08); // sum += ALPHA*A*B

                  vc9 = svmla_m(pg, vc9, vb8, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb9, alpha09); // sum += ALPHA*A*B

                  vc10 = svmla_m(pg, vc10, vb8, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb9, alpha010); // sum += ALPHA*A*B

                  vc11 = svmla_m(pg, vc11, vb8, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb9, alpha011); // sum += ALPHA*A*B

                  vc12 = svmla_m(pg, vc12, vb8, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb9, alpha012); // sum += ALPHA*A*B

                  vc13 = svmla_m(pg, vc13, vb8, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb9, alpha013); // sum += ALPHA*A*B

                  vc14 = svmla_m(pg, vc14, vb8, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb9, alpha014); // sum += ALPHA*A*B

                  vc15 = svmla_m(pg, vc15, vb8, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb9, alpha015); // sum += ALPHA*A*B
	
		alpha = A[i*lda+(k+2)];
	       	alpha0 = A[i*lda+(k+3)];

                alpha1 =  A[(i+1)*lda+(k+2)];
                alpha01 =  A[(i+1)*lda+(k+3)];

                 alpha2 =  A[(i+2)*lda+(k+2)];
                alpha02 =  A[(i+2)*lda+(k+3)];

                alpha3 =  A[(i+3)*lda+(k+2)];
                alpha03 =  A[(i+3)*lda+(k+3)];

                alpha4 =  A[(i+4)*lda+(k+2)];
                alpha04 =  A[(i+4)*lda+(k+3)];
               
                alpha5 = A[(i+5)*lda+(k+2)];
                alpha05 = A[(i+5)*lda+(k+3)];
 		
 		 alpha6 = A[(i+6)*lda+(k+2)];
 		 alpha06 = A[(i+6)*lda+(k+3)];
                
                 alpha7 =  A[(i+7)*lda+(k+2)];
                alpha07 =  A[(i+7)*lda+(k+3)];
                
                alpha8 =  A[(i+8)*lda+(k+2)];
                 alpha08 =  A[(i+8)*lda+(k+3)];
                
                alpha9 = A[(i+9)*lda+(k+2)];
                alpha09 = A[(i+9)*lda+(k+3)];
                
                alpha10 =  A[(i+10)*lda+(k+2)];
                alpha010 =  A[(i+10)*lda+(k+3)];
                
                alpha11 =  A[(i+11)*lda+(k+2)];
                alpha011 =  A[(i+11)*lda+(k+3)];
                
                 alpha12 =  A[(i+12)*lda+(k+2)];
                 alpha012 =  A[(i+12)*lda+(k+3)];
                
                 alpha13 =  A[(i+13)*lda+(k+2)];
                alpha013 =  A[(i+13)*lda+(k+3)];
                
                alpha14 = A[(i+14)*lda+(k+2)];
		alpha014 = A[(i+14)*lda+(k+3)];
                
                alpha15 =  A[(i+15)*lda+(k+2)];
                 alpha015 =  A[(i+15)*lda+(k+3)];

                 vc = svmla_m(pg,vc, vb10, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb11, alpha0); // sum += ALPHA*A*B

                  vc1 = svmla_m(pg, vc1, vb10, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb11, alpha01); // sum += ALPHA*A*B

                  vc2 = svmla_m(pg, vc2, vb10, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb11, alpha02); // sum += ALPHA*A*B

                  vc3 = svmla_m(pg, vc3, vb10, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb11, alpha03); // sum += ALPHA*A*B

                  vc4 = svmla_m(pg, vc4, vb10, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb11, alpha04); // sum += ALPHA*A*B

                  vc5 = svmla_m(pg, vc5, vb10, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb11, alpha05); // sum += ALPHA*A*B

                  vc6= svmla_m(pg, vc6, vb10, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb11, alpha06); // sum += ALPHA*A*B

                  vc7 = svmla_m(pg, vc7, vb10, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb11, alpha07); // sum += ALPHA*A*B

                  vc8 = svmla_m(pg, vc8, vb10, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb11, alpha08); // sum += ALPHA*A*B

                  vc9 = svmla_m(pg, vc9, vb10, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb11, alpha09); // sum += ALPHA*A*B

                  vc10 = svmla_m(pg, vc10, vb10, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb11, alpha010); // sum += ALPHA*A*B

                  vc11 = svmla_m(pg, vc11, vb10, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb11, alpha011); // sum += ALPHA*A*B

                  vc12 = svmla_m(pg, vc12, vb10, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb11, alpha012); // sum += ALPHA*A*B

                  vc13 = svmla_m(pg, vc13, vb10, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb11, alpha013); // sum += ALPHA*A*B

                  vc14 = svmla_m(pg, vc14, vb10, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb11, alpha014); // sum += ALPHA*A*B

                  vc15 = svmla_m(pg, vc15, vb10, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb11, alpha015); // sum += ALPHA*A*B


		////////////
		alpha = A[i*lda+(k+4)];
	       	alpha0 = A[i*lda+(k+5)];

                alpha1 =  A[(i+1)*lda+(k+4)];
                alpha01 =  A[(i+1)*lda+(k+5)];

                 alpha2 =  A[(i+2)*lda+(k+4)];
                alpha02 =  A[(i+2)*lda+(k+5)];

                alpha3 =  A[(i+3)*lda+(k+4)];
                alpha03 =  A[(i+3)*lda+(k+5)];

                alpha4 =  A[(i+4)*lda+(k+4)];
                alpha04 =  A[(i+4)*lda+(k+5)];
               
                alpha5 = A[(i+5)*lda+(k+4)];
                alpha05 = A[(i+5)*lda+(k+5)];
 		
 		 alpha6 = A[(i+6)*lda+(k+4)];
 		 alpha06 = A[(i+6)*lda+(k+5)];
                
                 alpha7 =  A[(i+7)*lda+(k+4)];
                alpha07 =  A[(i+7)*lda+(k+5)];
                
                alpha8 =  A[(i+8)*lda+(k+4)];
                 alpha08 =  A[(i+8)*lda+(k+5)];
                
                alpha9 = A[(i+9)*lda+(k+4)];
                alpha09 = A[(i+9)*lda+(k+5)];
                
                alpha10 =  A[(i+10)*lda+(k+4)];
                alpha010 =  A[(i+10)*lda+(k+5)];
                
                alpha11 =  A[(i+11)*lda+(k+4)];
                alpha011 =  A[(i+11)*lda+(k+5)];
                
                 alpha12 =  A[(i+12)*lda+(k+4)];
                 alpha012 =  A[(i+12)*lda+(k+5)];
                
                 alpha13 =  A[(i+13)*lda+(k+4)];
                alpha013 =  A[(i+13)*lda+(k+5)];
                
                alpha14 = A[(i+14)*lda+(k+4)];
		alpha014 = A[(i+14)*lda+(k+5)];
                
                alpha15 =  A[(i+15)*lda+(k+4)];
                 alpha015 =  A[(i+15)*lda+(k+5)];

                 vc = svmla_m(pg,vc, vb12, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb13, alpha0); // sum += ALPHA*A*B

                  vc1 = svmla_m(pg, vc1, vb12, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb13, alpha01); // sum += ALPHA*A*B

                  vc2 = svmla_m(pg, vc2, vb12, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb13, alpha02); // sum += ALPHA*A*B

                  vc3 = svmla_m(pg, vc3, vb12, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb13, alpha03); // sum += ALPHA*A*B

                  vc4 = svmla_m(pg, vc4, vb12, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb13, alpha04); // sum += ALPHA*A*B

                  vc5 = svmla_m(pg, vc5, vb12, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb13, alpha05); // sum += ALPHA*A*B

                  vc6= svmla_m(pg, vc6, vb12, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb13, alpha06); // sum += ALPHA*A*B

                  vc7 = svmla_m(pg, vc7, vb12, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb13, alpha07); // sum += ALPHA*A*B

                  vc8 = svmla_m(pg, vc8, vb12, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb13, alpha08); // sum += ALPHA*A*B

                  vc9 = svmla_m(pg, vc9, vb12, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb13, alpha09); // sum += ALPHA*A*B

                  vc10 = svmla_m(pg, vc10, vb12, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb13, alpha010); // sum += ALPHA*A*B

                  vc11 = svmla_m(pg, vc11, vb12, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb13, alpha011); // sum += ALPHA*A*B

                  vc12 = svmla_m(pg, vc12, vb12, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb13, alpha012); // sum += ALPHA*A*B

                  vc13 = svmla_m(pg, vc13, vb12, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb13, alpha013); // sum += ALPHA*A*B

                  vc14 = svmla_m(pg, vc14, vb12, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb13, alpha014); // sum += ALPHA*A*B

                  vc15 = svmla_m(pg, vc15, vb12, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb13, alpha015); // sum += ALPHA*A*B  
		///////////
		////unroll 8
		alpha = A[i*lda+(k+6)];
	       	alpha0 = A[i*lda+(k+7)];

                alpha1 =  A[(i+1)*lda+(k+6)];
                alpha01 =  A[(i+1)*lda+(k+7)];

                 alpha2 =  A[(i+2)*lda+(k+6)];
                alpha02 =  A[(i+2)*lda+(k+7)];

                alpha3 =  A[(i+3)*lda+(k+6)];
                alpha03 =  A[(i+3)*lda+(k+7)];

                alpha4 =  A[(i+4)*lda+(k+6)];
                alpha04 =  A[(i+4)*lda+(k+7)];
               
                alpha5 = A[(i+5)*lda+(k+6)];
                alpha05 = A[(i+5)*lda+(k+7)];
 		
 		 alpha6 = A[(i+6)*lda+(k+6)];
 		 alpha06 = A[(i+6)*lda+(k+7)];
                
                 alpha7 =  A[(i+7)*lda+(k+6)];
                alpha07 =  A[(i+7)*lda+(k+7)];
                
                alpha8 =  A[(i+8)*lda+(k+6)];
                 alpha08 =  A[(i+8)*lda+(k+7)];
                
                alpha9 = A[(i+9)*lda+(k+6)];
                alpha09 = A[(i+9)*lda+(k+7)];
                
                alpha10 =  A[(i+10)*lda+(k+6)];
                alpha010 =  A[(i+10)*lda+(k+7)];
                
                alpha11 =  A[(i+11)*lda+(k+6)];
                alpha011 =  A[(i+11)*lda+(k+7)];
                
                 alpha12 =  A[(i+12)*lda+(k+6)];
                 alpha012 =  A[(i+12)*lda+(k+7)];
                
                 alpha13 =  A[(i+13)*lda+(k+6)];
                alpha013 =  A[(i+13)*lda+(k+7)];
                
                alpha14 = A[(i+14)*lda+(k+6)];
		alpha014 = A[(i+14)*lda+(k+7)];
                
                alpha15 =  A[(i+15)*lda+(k+6)];
                 alpha015 =  A[(i+15)*lda+(k+7)];

                 vc = svmla_m(pg,vc, vb14, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb15, alpha0); // sum += ALPHA*A*B

                  vc1 = svmla_m(pg, vc1, vb14, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb15, alpha01); // sum += ALPHA*A*B

                  vc2 = svmla_m(pg, vc2, vb14, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb15, alpha02); // sum += ALPHA*A*B

                  vc3 = svmla_m(pg, vc3, vb14, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb15, alpha03); // sum += ALPHA*A*B

                  vc4 = svmla_m(pg, vc4, vb14, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb15, alpha04); // sum += ALPHA*A*B

                  vc5 = svmla_m(pg, vc5, vb14, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb15, alpha05); // sum += ALPHA*A*B

                  vc6= svmla_m(pg, vc6, vb14, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb15, alpha06); // sum += ALPHA*A*B

                  vc7 = svmla_m(pg, vc7, vb14, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb15, alpha07); // sum += ALPHA*A*B

                  vc8 = svmla_m(pg, vc8, vb14, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb15, alpha08); // sum += ALPHA*A*B

                  vc9 = svmla_m(pg, vc9, vb14, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb15, alpha09); // sum += ALPHA*A*B

                  vc10 = svmla_m(pg, vc10, vb14, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb15, alpha010); // sum += ALPHA*A*B

                  vc11 = svmla_m(pg, vc11, vb14, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb15, alpha011); // sum += ALPHA*A*B

                  vc12 = svmla_m(pg, vc12, vb14, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb15, alpha012); // sum += ALPHA*A*B

                  vc13 = svmla_m(pg, vc13, vb14, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb15, alpha013); // sum += ALPHA*A*B

                  vc14 = svmla_m(pg, vc14, vb14, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb15, alpha014); // sum += ALPHA*A*B

                  vc15 = svmla_m(pg, vc15, vb14, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb15, alpha015); // sum += ALPHA*A*B  
		/////unroll 8 end
			
		}
		else
		{
		
//		printf(" \nHi I am %d even ", flag);
	     
	       register float alpha = A[i*lda+k];
	       register float alpha0 = A[i*lda+(k+1)];
               
               register float alpha1 =  A[(i+1)*lda+k];
               register float alpha01 =  A[(i+1)*lda+(k+1)];
             
                register float alpha2 =  A[(i+2)*lda+k];
                register float alpha02 =  A[(i+2)*lda+(k+1)];
                
                register float alpha3 =  A[(i+3)*lda+k];
                register float alpha03 =  A[(i+3)*lda+(k+1)];
                
                register float alpha4 =  A[(i+4)*lda+k];
                register float alpha04 =  A[(i+4)*lda+(k+1)];
               
                register float alpha5 = A[(i+5)*lda+k];
                register float alpha05 = A[(i+5)*lda+(k+1)];
 		
 		register float alpha6 = A[(i+6)*lda+k];
 		register float alpha06 = A[(i+6)*lda+(k+1)];
               
                register float alpha7 =  A[(i+7)*lda+k];
                register float alpha07 =  A[(i+7)*lda+(k+1)];
                
                register float alpha8 =  A[(i+8)*lda+k];
                register float alpha08 =  A[(i+8)*lda+(k+1)];
                
                register float alpha9 = A[(i+9)*lda+k];
                register float alpha09 = A[(i+9)*lda+(k+1)];
               
                register float alpha10 =  A[(i+10)*lda+k];
                register float alpha010 =  A[(i+10)*lda+(k+1)];
               
                register float alpha11 =  A[(i+11)*lda+k];
                register float alpha011 =  A[(i+11)*lda+(k+1)];
                
                register float alpha12 =  A[(i+12)*lda+k];
                register float alpha012 =  A[(i+12)*lda+(k+1)];
                
                register float alpha13 =  A[(i+13)*lda+k];
                register float alpha013 =  A[(i+13)*lda+(k+1)];
                
                register float alpha14 = A[(i+14)*lda+k];
                register float alpha014 = A[(i+14)*lda+(k+1)];
                
                register float alpha15 =  A[(i+15)*lda+k];
                register float alpha015 =  A[(i+15)*lda+(k+1)];

		//for unroll4 uncomment below 
		vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb1, alpha0); // sum += ALPHA*A*B

                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb1, alpha01); // sum += ALPHA*A*B

                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb1, alpha02); // sum += ALPHA*A*B

                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb1, alpha03); // sum += ALPHA*A*B

                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb1, alpha04); // sum += ALPHA*A*B

                  vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb1, alpha05); // sum += ALPHA*A*B

                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb1, alpha06); // sum += ALPHA*A*B

                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb1, alpha07); // sum += ALPHA*A*B

                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb1, alpha08); // sum += ALPHA*A*B

                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb1, alpha09); // sum += ALPHA*A*B

                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb1, alpha010); // sum += ALPHA*A*B

                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb1, alpha011); // sum += ALPHA*A*B

                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb1, alpha012); // sum += ALPHA*A*B

                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb1, alpha013); // sum += ALPHA*A*B

                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb1, alpha014); // sum += ALPHA*A*B

                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb1, alpha015); // sum += ALPHA*A*B
	
		alpha = A[i*lda+(k+2)];
	       	alpha0 = A[i*lda+(k+3)];

                alpha1 =  A[(i+1)*lda+(k+2)];
                alpha01 =  A[(i+1)*lda+(k+3)];

                 alpha2 =  A[(i+2)*lda+(k+2)];
                alpha02 =  A[(i+2)*lda+(k+3)];

                alpha3 =  A[(i+3)*lda+(k+2)];
                alpha03 =  A[(i+3)*lda+(k+3)];

                alpha4 =  A[(i+4)*lda+(k+2)];
                alpha04 =  A[(i+4)*lda+(k+3)];
                
                alpha5 = A[(i+5)*lda+(k+2)];
                alpha05 = A[(i+5)*lda+(k+3)];
 		
 		 alpha6 = A[(i+6)*lda+(k+2)];
 		 alpha06 = A[(i+6)*lda+(k+3)];
                
                 alpha7 =  A[(i+7)*lda+(k+2)];
                alpha07 =  A[(i+7)*lda+(k+3)];
              
                alpha8 =  A[(i+8)*lda+(k+2)];
                 alpha08 =  A[(i+8)*lda+(k+3)];
               
                alpha9 = A[(i+9)*lda+(k+2)];
                alpha09 = A[(i+9)*lda+(k+3)];
               
                alpha10 =  A[(i+10)*lda+(k+2)];
                alpha010 =  A[(i+10)*lda+(k+3)];
                
                alpha11 =  A[(i+11)*lda+(k+2)];
                alpha011 =  A[(i+11)*lda+(k+3)];
                
                 alpha12 =  A[(i+12)*lda+(k+2)];
                 alpha012 =  A[(i+12)*lda+(k+3)];
                
                 alpha13 =  A[(i+13)*lda+(k+2)];
                alpha013 =  A[(i+13)*lda+(k+3)];
               
                alpha14 = A[(i+14)*lda+(k+2)];
		alpha014 = A[(i+14)*lda+(k+3)];
               
                alpha15 =  A[(i+15)*lda+(k+2)];
                 alpha015 =  A[(i+15)*lda+(k+3)];

                 vc = svmla_m(pg,vc, vb2, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb3, alpha0); // sum += ALPHA*A*B

                  vc1 = svmla_m(pg, vc1, vb2, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb3, alpha01); // sum += ALPHA*A*B

                  vc2 = svmla_m(pg, vc2, vb2, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb3, alpha02); // sum += ALPHA*A*B

                  vc3 = svmla_m(pg, vc3, vb2, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb3, alpha03); // sum += ALPHA*A*B

                  vc4 = svmla_m(pg, vc4, vb2, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb3, alpha04); // sum += ALPHA*A*B

                  vc5 = svmla_m(pg, vc5, vb2, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb3, alpha05); // sum += ALPHA*A*B

                  vc6= svmla_m(pg, vc6, vb2, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb3, alpha06); // sum += ALPHA*A*B

                  vc7 = svmla_m(pg, vc7, vb2, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb3, alpha07); // sum += ALPHA*A*B

                  vc8 = svmla_m(pg, vc8, vb2, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb3, alpha08); // sum += ALPHA*A*B

                  vc9 = svmla_m(pg, vc9, vb2, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb3, alpha09); // sum += ALPHA*A*B

                  vc10 = svmla_m(pg, vc10, vb2, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb3, alpha010); // sum += ALPHA*A*B

                  vc11 = svmla_m(pg, vc11, vb2, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb3, alpha011); // sum += ALPHA*A*B

                  vc12 = svmla_m(pg, vc12, vb2, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb3, alpha012); // sum += ALPHA*A*B

                  vc13 = svmla_m(pg, vc13, vb2, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb3, alpha013); // sum += ALPHA*A*B

                  vc14 = svmla_m(pg, vc14, vb2, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb3, alpha014); // sum += ALPHA*A*B

                  vc15 = svmla_m(pg, vc15, vb2, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb3, alpha015); // sum += ALPHA*A*B

		///////////unroll 6
		alpha = A[i*lda+(k+4)];
	       	alpha0 = A[i*lda+(k+5)];

                alpha1 =  A[(i+1)*lda+(k+4)];
                alpha01 =  A[(i+1)*lda+(k+5)];

                 alpha2 =  A[(i+2)*lda+(k+4)];
                alpha02 =  A[(i+2)*lda+(k+5)];

                alpha3 =  A[(i+3)*lda+(k+4)];
                alpha03 =  A[(i+3)*lda+(k+5)];

                alpha4 =  A[(i+4)*lda+(k+4)];
                alpha04 =  A[(i+4)*lda+(k+5)];
                
                alpha5 = A[(i+5)*lda+(k+4)];
                alpha05 = A[(i+5)*lda+(k+5)];
 		
 		 alpha6 = A[(i+6)*lda+(k+4)];
 		 alpha06 = A[(i+6)*lda+(k+5)];
                
                 alpha7 =  A[(i+7)*lda+(k+4)];
                alpha07 =  A[(i+7)*lda+(k+5)];
              
                alpha8 =  A[(i+8)*lda+(k+4)];
                 alpha08 =  A[(i+8)*lda+(k+5)];
               
                alpha9 = A[(i+9)*lda+(k+4)];
                alpha09 = A[(i+9)*lda+(k+5)];
               
                alpha10 =  A[(i+10)*lda+(k+4)];
                alpha010 =  A[(i+10)*lda+(k+5)];
                
                alpha11 =  A[(i+11)*lda+(k+4)];
                alpha011 =  A[(i+11)*lda+(k+5)];
                
                 alpha12 =  A[(i+12)*lda+(k+4)];
                 alpha012 =  A[(i+12)*lda+(k+5)];
                
                 alpha13 =  A[(i+13)*lda+(k+4)];
                alpha013 =  A[(i+13)*lda+(k+5)];
               
                alpha14 = A[(i+14)*lda+(k+4)];
		alpha014 = A[(i+14)*lda+(k+5)];
               
                alpha15 =  A[(i+15)*lda+(k+4)];
                 alpha015 =  A[(i+15)*lda+(k+5)];

                 vc = svmla_m(pg,vc, vb4, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb5, alpha0); // sum += ALPHA*A*B

                  vc1 = svmla_m(pg, vc1, vb4, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb5, alpha01); // sum += ALPHA*A*B

                  vc2 = svmla_m(pg, vc2, vb4, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb5, alpha02); // sum += ALPHA*A*B

                  vc3 = svmla_m(pg, vc3, vb4, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb5, alpha03); // sum += ALPHA*A*B

                  vc4 = svmla_m(pg, vc4, vb4, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb5, alpha04); // sum += ALPHA*A*B

                  vc5 = svmla_m(pg, vc5, vb4, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb5, alpha05); // sum += ALPHA*A*B

                  vc6= svmla_m(pg, vc6, vb4, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb5, alpha06); // sum += ALPHA*A*B

                  vc7 = svmla_m(pg, vc7, vb4, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb5, alpha07); // sum += ALPHA*A*B

                  vc8 = svmla_m(pg, vc8, vb4, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb5, alpha08); // sum += ALPHA*A*B

                  vc9 = svmla_m(pg, vc9, vb4, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb5, alpha09); // sum += ALPHA*A*B

                  vc10 = svmla_m(pg, vc10, vb4, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb5, alpha010); // sum += ALPHA*A*B

                  vc11 = svmla_m(pg, vc11, vb4, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb5, alpha011); // sum += ALPHA*A*B

                  vc12 = svmla_m(pg, vc12, vb4, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb5, alpha012); // sum += ALPHA*A*B

                  vc13 = svmla_m(pg, vc13, vb4, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb5, alpha013); // sum += ALPHA*A*B

                  vc14 = svmla_m(pg, vc14, vb4, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb5, alpha014); // sum += ALPHA*A*B

                  vc15 = svmla_m(pg, vc15, vb4, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb5, alpha015); // sum += ALPHA*A*B		

		//////unroll 6 end 
		/////// unroll 8 ///
		alpha = A[i*lda+(k+6)];
	       	alpha0 = A[i*lda+(k+7)];

                alpha1 =  A[(i+1)*lda+(k+6)];
                alpha01 =  A[(i+1)*lda+(k+7)];

                 alpha2 =  A[(i+2)*lda+(k+6)];
                alpha02 =  A[(i+2)*lda+(k+7)];

                alpha3 =  A[(i+3)*lda+(k+6)];
                alpha03 =  A[(i+3)*lda+(k+7)];

                alpha4 =  A[(i+4)*lda+(k+6)];
                alpha04 =  A[(i+4)*lda+(k+7)];
                
                alpha5 = A[(i+5)*lda+(k+6)];
                alpha05 = A[(i+5)*lda+(k+7)];
 		
 		 alpha6 = A[(i+6)*lda+(k+6)];
 		 alpha06 = A[(i+6)*lda+(k+7)];
                
                 alpha7 =  A[(i+7)*lda+(k+6)];
                alpha07 =  A[(i+7)*lda+(k+7)];
              
                alpha8 =  A[(i+8)*lda+(k+6)];
                 alpha08 =  A[(i+8)*lda+(k+7)];
               
                alpha9 = A[(i+9)*lda+(k+6)];
                alpha09 = A[(i+9)*lda+(k+7)];
               
                alpha10 =  A[(i+10)*lda+(k+6)];
                alpha010 =  A[(i+10)*lda+(k+7)];
                
                alpha11 =  A[(i+11)*lda+(k+6)];
                alpha011 =  A[(i+11)*lda+(k+7)];
                
                 alpha12 =  A[(i+12)*lda+(k+6)];
                 alpha012 =  A[(i+12)*lda+(k+7)];
                
                 alpha13 =  A[(i+13)*lda+(k+6)];
                alpha013 =  A[(i+13)*lda+(k+7)];
               
                alpha14 = A[(i+14)*lda+(k+6)];
		alpha014 = A[(i+14)*lda+(k+7)];
               
                alpha15 =  A[(i+15)*lda+(k+6)];
                 alpha015 =  A[(i+15)*lda+(k+7)];

                 vc = svmla_m(pg,vc, vb6, alpha); // sum += ALPHA*A*B
                 vc = svmla_m(pg,vc, vb7, alpha0); // sum += ALPHA*A*B

                  vc1 = svmla_m(pg, vc1, vb6, alpha1); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb7, alpha01); // sum += ALPHA*A*B

                  vc2 = svmla_m(pg, vc2, vb6, alpha2); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb7, alpha02); // sum += ALPHA*A*B

                  vc3 = svmla_m(pg, vc3, vb6, alpha3); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb7, alpha03); // sum += ALPHA*A*B

                  vc4 = svmla_m(pg, vc4, vb6, alpha4); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb7, alpha04); // sum += ALPHA*A*B

                  vc5 = svmla_m(pg, vc5, vb6, alpha5); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb7, alpha05); // sum += ALPHA*A*B

                  vc6= svmla_m(pg, vc6, vb6, alpha6); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb7, alpha06); // sum += ALPHA*A*B

                  vc7 = svmla_m(pg, vc7, vb6, alpha7); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb7, alpha07); // sum += ALPHA*A*B

                  vc8 = svmla_m(pg, vc8, vb6, alpha8); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb7, alpha08); // sum += ALPHA*A*B

                  vc9 = svmla_m(pg, vc9, vb6, alpha9); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb7, alpha09); // sum += ALPHA*A*B

                  vc10 = svmla_m(pg, vc10, vb6, alpha10); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb7, alpha010); // sum += ALPHA*A*B

                  vc11 = svmla_m(pg, vc11, vb6, alpha11); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb7, alpha011); // sum += ALPHA*A*B

                  vc12 = svmla_m(pg, vc12, vb6, alpha12); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb7, alpha012); // sum += ALPHA*A*B

                  vc13 = svmla_m(pg, vc13, vb6, alpha13); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb7, alpha013); // sum += ALPHA*A*B

                  vc14 = svmla_m(pg, vc14, vb6, alpha14); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb7, alpha014); // sum += ALPHA*A*B

                  vc15 = svmla_m(pg, vc15, vb6, alpha15); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb7, alpha015); // sum += ALPHA*A*B
		
		// unroll 8 end

		
		}		
		flag++;
		}}
		for ( int k1 = k; k1 < K; k1 += 1) {
                svfloat32_t vb = svld1(pg, &B[k1*ldb+j]);

                register float alpha = ALPHA * A[i*lda+k1];
                register float alpha1 = ALPHA * A[(i+1)*lda+k1];
                register float alpha2 = ALPHA * A[(i+2)*lda+k1];
                register float alpha3 = ALPHA * A[(i+3)*lda+k1];
                register float alpha4 = ALPHA * A[(i+4)*lda+k1];
                register float alpha5 = ALPHA * A[(i+5)*lda+k1];
                register float alpha6 = ALPHA * A[(i+6)*lda+k1];
                register float alpha7 = ALPHA * A[(i+7)*lda+k1];
                register float alpha8 = ALPHA * A[(i+8)*lda+k1];
                register float alpha9 = ALPHA * A[(i+9)*lda+k1];
                register float alpha10 = ALPHA * A[(i+10)*lda+k1];
                register float alpha11 = ALPHA * A[(i+11)*lda+k1];
                register float alpha12 = ALPHA * A[(i+12)*lda+k1];
                register float alpha13 = ALPHA * A[(i+13)*lda+k1];
                register float alpha14 = ALPHA * A[(i+14)*lda+k1];
                register float alpha15 = ALPHA * A[(i+15)*lda+k1];
                 vc = svmla_m(pg,vc, vb, alpha); // sum += ALPHA*A*B
                  vc1 = svmla_m(pg, vc1, vb, alpha1); // sum += ALPHA*A*B
                  vc2 = svmla_m(pg, vc2, vb, alpha2); // sum += ALPHA*A*B
                  vc3 = svmla_m(pg, vc3, vb, alpha3); // sum += ALPHA*A*B
                  vc4 = svmla_m(pg, vc4, vb, alpha4); // sum += ALPHA*A*B
                  vc5 = svmla_m(pg, vc5, vb, alpha5); // sum += ALPHA*A*B
                  vc6= svmla_m(pg, vc6, vb, alpha6); // sum += ALPHA*A*B
                  vc7 = svmla_m(pg, vc7, vb, alpha7); // sum += ALPHA*A*B
                  vc8 = svmla_m(pg, vc8, vb, alpha8); // sum += ALPHA*A*B
                  vc9 = svmla_m(pg, vc9, vb, alpha9); // sum += ALPHA*A*B
                  vc10 = svmla_m(pg, vc10, vb, alpha10); // sum += ALPHA*A*B
                  vc11 = svmla_m(pg, vc11, vb, alpha11); // sum += ALPHA*A*B
                  vc12 = svmla_m(pg, vc12, vb, alpha12); // sum += ALPHA*A*B
                  vc13 = svmla_m(pg, vc13, vb, alpha13); // sum += ALPHA*A*B
                  vc14 = svmla_m(pg, vc14, vb, alpha14); // sum += ALPHA*A*B
                  vc15 = svmla_m(pg, vc15, vb, alpha15); // sum += ALPHA*A*B
        }
                svst1(pg, &C[i*ldc+j], vc);
                svst1(pg, &C[(i+1)*ldc+j], vc1);
                svst1(pg, &C[(i+2)*ldc+j], vc2);
                svst1(pg, &C[(i+3)*ldc+j], vc3);
                svst1(pg, &C[(i+4)*ldc+j], vc4);
                svst1(pg, &C[(i+5)*ldc+j], vc5);
                svst1(pg, &C[(i+6)*ldc+j], vc6);
                svst1(pg, &C[(i+7)*ldc+j], vc7);
                svst1(pg, &C[(i+8)*ldc+j], vc8);
                svst1(pg, &C[(i+9)*ldc+j], vc9);
                svst1(pg, &C[(i+10)*ldc+j], vc10);
                svst1(pg, &C[(i+11)*ldc+j], vc11);
                svst1(pg, &C[(i+12)*ldc+j], vc12);
                svst1(pg, &C[(i+13)*ldc+j], vc13);
                svst1(pg, &C[(i+14)*ldc+j], vc14);
                svst1(pg, &C[(i+15)*ldc+j], vc15);
        }
        }//}

//int k_left = k;
  int i_left=i;
    #pragma omp for
  for ( j = jj; j < N; j+= svcntw()) {
     svbool_t pg = svwhilelt_b32(j, N);
//printf("\nthread id=%d  each thread N=%d,id j=%d", omp_get_thread_num(), N,j);
     svfloat32_t  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     for (i=i_left; i < M; i += 4) {
        vc= svld1(pg, &C[i*ldc+j]);
        if((i+1) < M)  {vc1= svld1(pg, &C[(i+1)*ldc+j]);}
        if ((i+2) < M) {vc2= svld1(pg, &C[(i+2)*ldc+j]);}
        if ((i+3) < M) {vc3= svld1(pg, &C[(i+3)*ldc+j]);}
       for (int k = kk; k < K; k += 1) {
                alpha = ALPHA * A[i*lda+k];
                if ((i+1) < M) {alpha1 = ALPHA * A[(i+1)*lda+k]; }
                if ((i+2) < M) { alpha2 = ALPHA * A[(i+2)*lda+k];}
                if ((i+3) < M) { alpha3 = ALPHA * A[(i+3)*lda+k];}
                vb = svld1(pg, &B[k*ldb+j]);
                  vc = svmla_m(pg, vc, vb, alpha); // sum += ALPHA*A*B
                  if ((i+1) < M) {vc1 = svmla_m(pg, vc1, vb, alpha1);} // sum += ALPHA*A*B
                  if ((i+2) < M) {vc2 = svmla_m(pg, vc2, vb, alpha2);} // sum += ALPHA*A*B
                  if ((i+3) < M) {vc3 = svmla_m(pg, vc3, vb, alpha3);}// sum += ALPHA*A*B
        }
          svst1(pg, &C[i*ldc+j], vc);
          if ((i+1) < M)      {svst1(pg, &C[(i+1)*ldc+j], vc1);}
          if ((i+2) < M)      {svst1(pg, &C[(i+2)*ldc+j], vc2);}
          if ((i+3) < M)      {svst1(pg, &C[(i+3)*ldc+j], vc3);}
     }
  }}
}
/********* unroll 16 k 16 end*******/





void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
	//printf("clang in am in gemm");
//    printf("M, N, K, lda, ldb, ldc: %d %d %d %d %d %d\n", M, N, K, lda, ldb, ldc);
    int i, j;
if(BETA != 1.0)
{
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
}
    if(!TA && !TB)
	{
	   float *transposeB, *transposeA;
	   //float transposeB[16*4096*128], transposeA[16*4096*128];

	//lda = M;
	//int blockM = ((16 >M)?M:(16)) ;  ///original
	//int blockM = ((48 >M)?M:(48)) ; 
	//int blockM = ((192 >M)?M:(192)) ; 
	//int blockM = ((192 >M)?M:(192)) ; 
	int blockM = ((64 >M)?M:(64)) ; //from 2 - 12 threads
	//int blockM = ((16 >M)?M:(16)) ; 
	//int blockM = ((512 >M)?M:(512)) ; 
//	int blockM = ((128 >M)?M:(128)) ; ///48 cores
	//int blockM = ((256 >M)?M:(256)) ; ///48 cores
	//int blockM = ((1024 >M)?M:(1024)) ; //48
//	int blockM = ((1024 >M)?M:(1024)) ; 
//	int blockN  =((1024>N)?N:(1024));
//	int blockN  =((2048>N)?N:(2048));
//	int blockN  =((4096>N)?N:(4096));     ///original
	//int blockN  =((7701>N)?N:(7701));     ///original
	//int blockN  =((8192>N)?N:(8192)); //48
	int blockN  =((16384>N)?N:(16384)); //48
	//int blockN  =((64>N)?N:(64));
	//int blockK = ((64>K)?K:(64));
	//int blockK = ((1024>K)?K:(1024)); //48
	int blockK = ((2048>K)?K:(2048));
	//int blockK = ((4096>K)?K:(4096));
//	int blockK = ((128>K)?K:(128));      ///original
	//int blockK = ((256>K)?K:(256));    //from 2-12 threads //48 cores
	//int blockK = ((512>K)?K:(512));
        transposeB= (float *)malloc(blockM*blockN*blockK*sizeof(float));
        transposeA= (float *)malloc(blockM*blockN*blockK*sizeof(float));
        //transposeB= (float *)calloc(blockM*blockN*blockK, sizeof(float));
        //transposeA= (float *)calloc(blockM*blockN*blockK , sizeof(float));
        //transposeB= (float *)malloc(64*4096*256*sizeof(float));
        //transposeA= (float *)malloc(64*4096*256*sizeof(float));
	
	if (transposeB == NULL) {
        fprintf(stderr, "Fatal: failed to allocate bytes.\n");
        exit(0);
    	}
	if(transposeA == NULL) {
        fprintf(stderr, "Fatal: failed to allocate  bytes.\n");
       exit(0);
    	}
	//printf("blockM, blockN, blockK, M, N, K = %d, %d, %d %d %d %d\n", blockM, blockN, blockK, M, N, K);
	//int blockK = ((512>K)?K:(512));
	//gemm_nn_pack(M, N, K, ALPHA,A, lda, B, ldb,C, ldc, blockM, blockN, blockK);
	//gemm_nn_pack1(M, N, K, ALPHA,A, lda, B, ldb,C, ldc, blockM, blockN, blockK); // working and giving best one 
	gemm_nn_pack2(M, N, K, ALPHA,A, lda, B, ldb,C, ldc, blockM, blockN, blockK, transposeB, transposeA);  ///// MAIN FUNCTION
	//gemm_nn1_transpose(M, N, K, ALPHA,A, lda, B, ldb,C,transpose, ldc, blockM, blockN, blockK);  // uncomment to run unroll 16 + K 4 + double buffer
	//gemm_nn_unroll16k8_doublebuffer(0,0,0,M, N, K, ALPHA,A,lda, B, ldb,C,ldc); //utilize full 32 registers
	//gemm_nn1(0,0,0,M, N, K, ALPHA,A,lda, B, ldb,C,ldc);  // uncomment to run unroll 16 + K 4 + double buffer
	//gemm_nn_unroll16(0,0,0,M, N, K, ALPHA,A,lda, B, ldb,C,ldc); // uncomment to run unroll 16 	
        //gemm_nn_unroll16_noalpha(0,0,0,M, N, K, ALPHA,A,lda, B, ldb,C,ldc); 
	//gemm_nn1_unroll24(0,0,0,M, N, K, ALPHA,A,lda, B, ldb,C,ldc);  // uncomment to unroll 24
	//gemm_nn1_unroll24_alpha(0,0,0,M, N, K, ALPHA,A,lda, B, ldb,C,ldc);  // uncomment to unroll 24 with alpha
	if(transposeB != NULL)
	{
		free(transposeB);
		transposeB = NULL;
	}
	if(transposeA != NULL)
	{
		free(transposeA);
		transposeA = NULL;
	}
	}
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

