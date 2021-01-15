#include <iostream>
#include <typeinfo>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include "SuiteSparseQR.hpp"

#include "cholmod_core.h"
#include "cholmod_internal.h"

using namespace std;


int main(int argc, char **argv) {
  cholmod_common Common, *cc;
  cholmod_sparse *A;
  cholmod_dense *X, *B, *Residual;
  double rnorm, one [2] = {1, 0}, minusone [2] = {-1, 0};
  int mtype;
  double *pt, *ptb;

  //start cholmod
  cc = &Common;
  cholmod_l_start(cc);
  
  //load A 
  cout << "pleae input the matrix market file" << endl;
  FILE *mfile;
  mfile = fopen("/home/robert/projects/SPQR_TEST/build/m_05_05_crg.mm", "r");
  A = (cholmod_sparse *) cholmod_l_read_matrix(mfile, 1, &mtype, cc);
  cout << "input the A matrix over" << endl;
  fclose(mfile);
  //B = ones(size(A, 1), 1)
  B = cholmod_l_ones (A->nrow, 1, A->xtype, cc);
  cout << "B vector is: " << B->nzmax << endl;
  //X = A\B
  X = SuiteSparseQR <double> (A, B, cc);
  
  //rnorm = norm(B-A*X)
  Residual = cholmod_l_copy_dense(B, cc);
  cout << "Residual type is: " << Residual->dtype << endl;
  cholmod_l_sdmult(A, 0, minusone, one, X, Residual, cc);
  cout << "sdmult over" << endl;
  rnorm = cholmod_l_norm_dense (Residual, 2, cc);
  printf ("2-norm of residual: %8.1e\n", rnorm);
  printf ("rank %ld\n", cc->SPQR_istat[4]);
  
  cout << "B:" << B->nrow << endl;
  cout << "A:" << A->nzmax << endl;
  cout << "X nrow:" << X->nrow << "\nX ncol:" << X->ncol << 
  "\nX nzmax:" << X->nzmax << "\nX xtype:" << (X->x)  << "\nX x:" << X->x << "\nX z:" << X->z << endl;
   
  pt =(double*) X->x;
  int countpt = X->nzmax;
  while (countpt){
      cout << *pt << endl;
      pt++;
      countpt--;
  }
  ptb = (double*) B->x;
  countpt = B->nzmax;
  while (countpt){
      cout << "start b" << endl;
      cout << *ptb << endl;
      ptb++;
      countpt--;
  }
  //free everything and finish CHOLMOD
  cholmod_l_free_dense (&Residual, cc);
  cout << "free dense Residual over" << endl;
  cholmod_l_free_sparse (&A, cc);
  cout << "free sparse over" << endl;
  cholmod_l_free_dense (&X, cc);
  cout << "free dense X over" << endl;
  cholmod_l_free_dense (&B, cc);
  cout << "free dense B over" << endl;
  cholmod_l_finish (cc);
  cout << "finish cc over" << endl;  
  
  /*Eigen::Matrix3d A;
  Eigen::Vector3d b,x;
  A << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  b << 1, 2, 3;
  Eigen::SPQR<Eigen::Matrix3d > solver;
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {
      return 0;
  }
  x = solver.solve(b);
  if(solver.info() != Eigen::Success) {
      return 0;
  }
  cout << "the result is: " << x <<endl;*/
  cout << "Hello, world1!" <<endl;
  
  return 0;          
}
