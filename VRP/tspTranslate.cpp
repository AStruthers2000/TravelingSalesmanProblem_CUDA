
#include <iostream>
#include <string>
#include <string.h>
#include <fstream>
#include <sstream>
#include <math.h>
using namespace std;

/**
 * @brief A method for creating an adjacency matrix from a .tsp file containing a list of coordinates. Method will use these points to create an adjacency matrix representing
 *  the euclidean distance between these points.
 * 
 * @param matrix The starting address of a flattended matrix to be populated with the distances between points in the .tsp file. It is assumed that this matrix has 
 * allocated enough memory to hold the values in the given .tsp file.
 * @param location The location of the local .tsp file holding the problem data.
 * @param problemSize The size of the problem being solved. (i.e. the number of poitns in the problem.)
 */

void getSTSPMatrix(double* matrix, string location, int problemSize){

    // create and open file stream
    ifstream tsp;
    tsp.open(location, ios::in);

    // create value to hold the line currently being read.
    string line;

    // create an integer i to indicate the value of the node being read.
    int i = 0;

    // create an array and allocate memory to hold the location of each point
    double* pointsLocations = (double*) malloc(sizeof(double) * problemSize * 2);

    // for every line in the .tsp data file.
    while(getline(tsp, line)){
        
        // if the line is the end of the file, end the loop
        if(line == "EOF") break;
        
        // create a string stream to read the values in the line.
        stringstream ssm(line);

        // create and read values on each line.
        int index = 0;
        double x = 0;
        double y = 0;

        if(ssm >> index >> x >> y) {}else{continue;} // if there is an error reading the line, continue to the next iteration.

        // store the x and y coordinates in the pointsLocations matrix
        pointsLocations[i] = x;
        pointsLocations[i + 1] = y;
        
        // incriment i by 2.
        i+=2;
    }

    // using pointsLocations, create an adjacency matrix by euclidian distance

    // for every point
    for(i = 0; i < problemSize*2; i+=2){

        // get the x and y values of the curent point.
        double origin_x = 0;
        double origin_y = 0;

        origin_x = pointsLocations[i];
        origin_y = pointsLocations[i+1];

        // for every other point
        for(int j = 0; j < problemSize*2; j+=2){

            // get the coordinates of the other point
            double second_x = 0;
            double second_y = 0;

            second_x = pointsLocations[j];
            second_y = pointsLocations[j+1];

            // calculate the euclidean distance between the points and save it in the adjacency matrix.
            matrix[(i/2)*problemSize +  (j/2)] = sqrt(pow(abs(second_x - origin_x), 2) + pow(abs(second_y - origin_y), 2));
        }
    }

    // close point and free memory
    tsp.close();
    free(pointsLocations);

}

/**
 * @brief Populates a given integer array with values read from a local .atsp file of adjacency matrix data.
 * 
 * @param matrix The starting address of an array. It is assumed that the given address has allocated enough memory to store data int he .atsp file.
 * @param location The local file location of the .atsp file.
 * @param nullKey The key that represents a value of 0 in the atsp data.
 */
void getATSPMatrix(int* matrix, string location, int nullKey){

    // create and open the file stream to read ATSP data
    ifstream tsp;
    tsp.open(location, ios::in);

    // line to hold the curent line being read
    string line;

    // the curent index of data being accessed and assigned in the matrix and file.
    int i = 0;

    // for every line in the document
    while(getline(tsp, line)){

        // create a string stream to read the values in the curent line.
        stringstream ssm(line);

        // create a temporary int to hold the value currently being accessed in the file.
        int cur = 0;
        
        // for every integer in the line
        while(ssm >> cur){
            if(cur == nullKey){ // the the curent node is being evaluated save 0 at i
                matrix[i] = 0;
            }
            else{   // otherwise save the value
                matrix[i] = cur;
            }

            i++;
        }
    }
}

// helper function to print the ATSP adjacency matrix.
void printMatrix(int* matrix, int problemSize){

    for(int i = 0; i < problemSize; i++){
        for(int j = 0; j < problemSize; j++){

            cout << matrix[i*problemSize + j] << " ";
        }
        cout << endl;
    }
}

// helper function to print the STSP adjacency matrix.
void printMatrix(double* matrix, int problemSize){
    for(int i = 0; i < problemSize; i++){
        for(int j = 0; j < problemSize; j++){

            cout << matrix[i*problemSize + j] << " ";
        }
        cout << endl;
    }
}



int main(){
    const int STSPproblemSize = 1400;

    double* stspMatrix = (double*) malloc(sizeof(double) * STSPproblemSize * STSPproblemSize);

    string stspLocation = "fl1400.tsp";


    const int ATSPproblemSize = 65;

    int* atspMatrix = (int*) malloc(sizeof(int) * ATSPproblemSize * ATSPproblemSize);

    string atspLocation = "ftv64.atsp";


    getSTSPMatrix(stspMatrix, stspLocation, STSPproblemSize);
    getATSPMatrix(atspMatrix, atspLocation, 100000000);


    printMatrix(stspMatrix, STSPproblemSize);
    printMatrix(atspMatrix, ATSPproblemSize);

    


    free(atspMatrix);
    free(stspMatrix);
}