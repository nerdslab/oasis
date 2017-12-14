/** oASIS: Adaptive Column Sampling for Kernel Matrix Approximation
 * R. Patel, T. Goldstein, E. Dyer, A. Mirhoseini, and R. Baraniuk
 * Submitted to IEEE JSTSP
 */

/**Readme file for oASIS_P.cpp, used in experiments for Kernel Matrix 
 *Approximation. 
 */


If you don't have eigen, download it:
http://eigen.tuxfamily.org/index.php?title=Main_Page

Then, in terminal, set an eigen path:
EXPORT EIGEN=path to eigen

-----------------------------------
Compile oASIS-P by:
mpic++ oASIS_P.cpp -o oasisp.out -I$EIGEN

General Use:
#mpiexec -n numprocessors exefilename datafile n m num_start Lmin Lskip Lmax Sigma

You can test it:
mpiexec -n 2 ./oasisp.out matrix.txt 2000 2 1 20 20 100 0.1