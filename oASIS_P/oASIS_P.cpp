/** oASIS: Adaptive Column Sampling for Kernel Matrix Approximation
 * R. Patel, T. Goldstein, E. Dyer, A. Mirhoseini, and R. Baraniuk
 * Submitted to IEEE JSTSP
 */

/**The MIT License (MIT)
 
 * Copyright (c) 2014 Rice University
 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * Implimentation of OASIS-P algorithm with a Gaussian Kernel on a cluster
 * [m n] = size of matrix A, where A is the data arranged columnwise (each point of data is a column of A)
 * k = number of columns to be selected; [n k] = size of matrix C
 * s = number of initially randomly selected columns
 * sigma = sigma of the Gaussian kernel exp(-(||ai-aj||_2)^2/sigma)
 * Code written by Azalia Mirhoseini and Raajen Patel 
 */


#include <mpi.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#define err_countMax 100000 // maximum number of columns chosen for error analysis per core

using namespace std;
using namespace Eigen;


//DEFINE KERNEL HERE - currently using Gaussian Kernel. Change for other kernels.
double kernelf(const Ref<const MatrixXd>& a1, const Ref<const MatrixXd>& a2,  double sig)
{
	double temp;
	
	temp = exp(-((a1-a2).norm()*(a1-a2).norm())/sig);
	return temp;
}   
//END OF KERNEL DEFINITIONS


//Find the core that has the selected column index
uint64_t getRank(uint64_t idx, int npes, int n)
{
    uint64_t myn = (n+(npes-1))/npes;
    return idx / myn;
}


int main(int argc, char *argv[])
{
	
	int npes, myrank;
	double t1, t2, tread1, tread2;
	
    //Initialize MPI protocol
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	srand(time(0) + myrank);
	
    //error checking for inputs
	if (argc<9)
	{
		if(!myrank)
			cout << "Please enter path to matrix A, n,  m,  s, lmin, lskip, lmax, sigma"<<endl;
		MPI_Finalize();
		return -1;
	}
	
    
	uint64_t n = atoi(argv[2]);
	uint64_t m = atoi(argv[3]);
	int s = atoi(argv[4]);
	int lmin = atoi(argv[5]);
	int lskip = atoi(argv[6]);
	int k = atoi(argv[7]);
	int init_s = s;
	double sig = atof(argv[8]);
	
    //error checking for inputs
	if (s>k)
	{
		if(!myrank)
			cout << " s (number of initially randomly selected columns) should be less than k (total number of selected columns)"<<endl;
		MPI_Finalize();
		return -1;
	}
	
    //error checking for inputs
	if (lskip<1)
	{
		if(!myrank)
			cout << " lskip should be at least 1."<<endl;
		MPI_Finalize();
		return -1;
	}
	
    //error checking for inputs
	if (sig==0)
	{
		if(!myrank)
			cout << "Sigma needs to be greater than 0."<<endl;
		MPI_Finalize();
		return -1;
	}
	
    //error checking for inputs
	uint64_t myn = (n+(npes-1))/npes;
	uint64_t mystartid= myrank*myn;
	MPI_Barrier(MPI_COMM_WORLD);
	int neg = (n-((npes-1)*myn));
	if (neg<0)
	{
		if(!myrank)
			cout << "Process Terminated: Increase n or Reduce ncpu"<<endl;
		MPI_Finalize();
		return -2;
	}
	
	
	
	if(myrank==npes-1)
	{
		myn = n - mystartid;  
	}
	
	MatrixXd A(m,myn);  
	MatrixXd C(myn,k);
	MatrixXd R(k,myn);
	MatrixXd w(k,k);
	MatrixXd invw(k,k);
	MatrixXd d(myn,1);
	MatrixXd delta(myn,1);
	
	if(!myrank)
	{
		printf( "\033[32;1mStart loading A from file\033[0m\n");
	}
	
	
	tread1 = MPI_Wtime(); 
	ifstream fr;
	fr.open(argv[1]);
	
	//Read in the matrix, in .txt format. Change if you have a different read in format.
    //Note that each node gets a "block" of the data matrix, consisting of a subset of
    //complete points. 
	int root;
	double temp;
	
	for(uint64_t i=0; i< m; i++)
	{
		for(uint64_t j=0; j<n; j++)   
		{
			fr >> temp;			
			root = getRank(j,npes, n);
			
			if(myrank==root)
			{
				A(i, j-mystartid) = temp;
			}
		}
	}		
	
	fr.close();
	
	tread2 = MPI_Wtime(); 
		
	
	if(!myrank)
	{
		printf( "\033[32;1mDone loading A from file\033[0m\n");
	}
	
	if(!myrank)
		printf( "\033[1;31mMatrix A loading time = %fs\033[0m\n",(tread2 - tread1));
	
	
	
	
	/*compute vector of diagonal entries*/
	for(uint64_t i=0; i< myn; i++)
	{
		d(i,0) = kernelf(A.col(i),A.col(i),sig);  
	}
	
	/*initialize delta*/
	for(uint64_t i=0; i< myn; i++)
	{
		delta(i,0) = 0;
	}
	
	
	/*create w by randomly selecting #s input indices*/
	uint64_t s_ind[k]; 
	uint64_t range  = n;
	
	if(!myrank)
	{
		for(int i=0; i< s; i++)
		{
			s_ind[i] = rand()%n; //This can be changed to read specific indices if desired.
		}
	}
	
	MPI_Bcast(s_ind, s, MPI_UINT64_T, 0, MPI_COMM_WORLD);  
	MPI_Barrier(MPI_COMM_WORLD);
	
	int root_i;
	int root_j;
	MatrixXd Acol_i(m,1);
	MatrixXd Acol_j(m,1);
	
	for(int i=0; i< s; i++)
	{
		root_i = getRank(s_ind[i],npes, n);
		
		if(myrank==root_i)
        {
			int ii = s_ind[i]-mystartid;
			Acol_i = A.col(ii); 
		}
		
		MPI_Bcast(Acol_i.data(), Acol_i.size(), MPI_DOUBLE, root_i, MPI_COMM_WORLD);  
		
		for(int j=0; j< s; j++)
		{
			root_j = getRank(s_ind[j],npes, n);
			
			if(myrank==root_j)
			{
				int jj = s_ind[j]-mystartid; 
				Acol_j = A.col(jj);
			}
			
			MPI_Bcast(Acol_j.data(), Acol_j.size(), MPI_DOUBLE, root_j, MPI_COMM_WORLD);  
			
			w(i,j)= kernelf(Acol_i, Acol_j,sig);  
		}
	}
	
	invw.block(0,0,s,s) = w.block(0,0,s,s);
	invw.block(0,0,s,s) = invw.block(0,0,s,s).inverse();
		
    
    
	/*compute local C_s*/
	for(int j=0; j< s; j++)
	{
		root_j = getRank(s_ind[j],npes, n);
		
		if(myrank==root_j)
		{
			int jj = s_ind[j]-mystartid; 
			Acol_j = A.col(jj);
		}
		
		MPI_Bcast(Acol_j.data(), Acol_j.size(), MPI_DOUBLE, root_j, MPI_COMM_WORLD);  

		for(uint64_t i = 0; i < myn ; i++)
		{
			C(i,j)= kernelf((A.col(i)),(Acol_j),sig);   
		} 
	}
	
    
    
	/*compute local R_s*/
	R.block(0,0,s,myn) = invw.block(0,0,s,s)*(C.block(0,0,myn,s).transpose());
	
	double maxval;
	double maxval_all;
	uint64_t  maxid_all;
	uint64_t maxid , in_s;
	MatrixXd qk(k,1);
	MatrixXd Acol_s(m,1);
	int root_s;

	
	struct maxidval
	{
		double val;
		int rank;
	};
	
	maxidval buff, out;
	
	MatrixXd v;
	int kk;
	
	t1 = MPI_Wtime();
    
    
    
	/* *****start selecting columns***** */
	if(!myrank)
	{cout<<"Start selecting columns"<<endl;
	}
	while (s<k)
	{
		/*compute delta*/
		for(uint64_t i=0; i< myn; i++)
		{
			delta(i,0) =  (abs(d(i,0)-(C.block(0,0,myn,s).row(i)*R.block(0,0,s,myn).col(i))) ) / d(i,0); //Delta computation per node
		}
		
        /*find max delta*/
		maxval = 0;
		
		maxid = 0;
		in_s = 0;
		vector<uint64_t> s_indSorted (s_ind,s_ind+s);  
		
		sort (s_indSorted.begin(), s_indSorted.begin()+s);  
		
		uint64_t i =0; 
		kk = 0;
		
        /* Find the largest delta and send the respective data point to everyone */
		while(i< myn)
		{
			if(i+mystartid==s_indSorted[kk])
			{
				kk++;
			}
			else if(maxval<delta(i,0))
			{
				maxval = delta(i,0);
				maxid = i+mystartid;
			}
			i++;
		}

		buff.rank = myrank;     
		buff.val = maxval;
				
		MPI_Barrier(MPI_COMM_WORLD);
		
		MPI_Allreduce(&buff, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		
		maxval_all = out.val;
		if(myrank==out.rank)
			maxid_all = maxid;
		
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&maxid_all, 1, MPI_UINT64_T, out.rank, MPI_COMM_WORLD);  
		MPI_Barrier(MPI_COMM_WORLD);
		s_ind[s]= maxid_all;
		
		int root_s;
		root_s = getRank(s_ind[s],npes, n);
		
		if(myrank==root_s)
		{
			int ss = s_ind[s]-mystartid; 
			Acol_s = A.col(ss);
		}
		
		MPI_Bcast(Acol_s.data(), Acol_s.size(), MPI_DOUBLE, root_s, MPI_COMM_WORLD);  
		
		
		
		/*update C */
		for( uint64_t i = 0; i < myn ; i++){
			C(i,s) = kernelf(A.col(i),(Acol_s),sig);  
		}
		
		double sk = 1./maxval_all;	 
		
		for(int i = 0; i < s ; i++){
			
			root_i = getRank(s_ind[i],npes, n);

			if(myrank==root_i)
			{
				int ii = s_ind[i]-mystartid; 
				Acol_i = A.col(ii);           
			}
			
			MPI_Bcast(Acol_i.data(), Acol_i.size(), MPI_DOUBLE, root_i, MPI_COMM_WORLD);
			qk(i) = kernelf((Acol_i),(Acol_s),sig);  		
		}
		
		qk.block(0,0,s,1) = invw.block(0,0,s,s)*qk.block(0,0,s,1);
		
		invw.block(0,0,s,s) = invw.block(0,0,s,s)+ sk* (qk.block(0,0,s,1)*(qk.block(0,0,s,1).transpose()));
		invw.block(0,s,s,1) = -sk*qk.block(0,0,s,1);
		invw.block(s,0,1,s) = -sk*(qk.block(0,0,s,1).transpose());
		invw(s,s) = sk;			
		
		
		
		/*update R */
		R.block(s,0,1,myn) = -sk*((qk.block(0,0,s,1).transpose())*(C.block(0,0,myn,s).transpose()) - C.block(0,s,myn,1).transpose()); 
		R.block(0,0,s,myn) =   R.block(0,0,s,myn) - (qk.block(0,0,s,1)*R.block(s,0,1,myn));
		
		
		/* new iteration*/
		s++;
		
		/* progress report (%)*/
		//if(!myrank && s%lskip==0)
		//{	
		//	int prcnt = (s*100)/k;
		//	printf( "\033[34;1m %%%d complete\033[0m\n",prcnt);
		//}		
		

		/* *****error computation***** */
		if(s>=lmin && (s-lmin)%lskip==0)
		{
			
			double ts1,ts2;
			ts1 =  MPI_Wtime();
			vector<uint64_t> s_indSorted (s_ind,s_ind+s); 
			sort (s_indSorted.begin(), s_indSorted.begin()+s);  
			ts2 =  MPI_Wtime();
			
			
			ts1 =  MPI_Wtime();
			
			
			int numerr= fmin(err_countMax/npes, myn);
			
			int err_ind_i, err_ind_j;
			
			double sumerr [2];
			sumerr[0] = 0;
			sumerr[1] = 0;
			double finalerr[2];
			double tempa;
			double  tempb;
			double sumnormg=0;
			for(int i=0; i< numerr; i++)
			{   
				err_ind_i= rand()%myn; 
				err_ind_j = rand()%myn; 
				
				tempa = kernelf(A.col(err_ind_i),A.col(err_ind_j),sig);  
				tempb = C.row(err_ind_i)*invw*(C.row(err_ind_j).transpose());
				sumerr [0] =  sumerr[0]+(tempa- tempb)*(tempa- tempb); // norm 2 error
				sumerr[1]  = sumerr[1]+ abs((tempa- tempb)); // norm 1 error
				sumnormg = sumnormg+tempa*tempa;
			}						
			sumerr[0] = sumerr[0]/numerr;	
			sumerr[1] = sumerr[1]/sumnormg;
			
			if( isnan(sumerr[0]))
			{
				sumerr[0] = 0;
			}
			if( isnan(sumerr[1]))
			{
				sumerr[1] = 0;
			}
			
			MPI_Barrier(MPI_COMM_WORLD);								
			MPI_Allreduce(sumerr, finalerr, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
						
			
			ts2 =  MPI_Wtime();	
			if(!myrank)
			{
				finalerr[0] =  finalerr[0]/npes;
				finalerr[1] =  finalerr[1]/npes;
				printf("cols = %d, Average l2 Error =  %e, Average l1 Error = %e\n", s, finalerr[0], finalerr[1]);
				
				if(s==k)
				{
					cout<<"Indices of selected columns="<<endl;
					for(int i=0; i<s; i++){      
						cout<<s_ind[i]<<" ";
					}
					cout<<endl;
				}
			}
			
		}
		
	}
        
        
	/* clean up */
	MPI_Barrier(MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	
	/*time elapsed*/
	if(!myrank)
	{
		printf( "\033[1;31mTotal time elapsed = %fs\033[0m\n",(t2 - tread1) );
	}
	
	
	
	
	MPI_Finalize();
	
	
	return 0;
}
