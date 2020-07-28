// functions and methods for jackknife analysis and multivariable function fits for correlated data sets

// used headers/libraries
#include <Eigen/Dense>
#include <gsl/gsl_cdf.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <numeric>

// ------------------------------------------------------------------------------------------------------------

//
//
// READING GIVEN DATASET FOR FURTHER ANALYSIS
//
//

// read file with dataset into a raw matrix
Eigen::MatrixXd ReadFile(std::string const &fileName)
{
    // start reading
    std::ifstream fileToRead;
    fileToRead.open(fileName);

    // determine number of columns
    std::string firstLine;
    std::getline(fileToRead, firstLine);
    std::stringstream firstLineStream(firstLine);

    // number of columns in given file
    int numOfCols = 0;
    std::string tmpString;
    // count number of writes to a temporary string container
    while (firstLineStream >> tmpString)
    {
        numOfCols++;
    }
    fileToRead.close();

    // string for all the lines
    std::string line;

    // data structure (raw matrix) to store data
    Eigen::MatrixXd rawDataMat(0, numOfCols);

    // reopen file
    fileToRead.open(fileName);
    // check if open
    if (fileToRead.is_open())
    {
        // read line by line
        int i = 0;
        while (std::getline(fileToRead, line))
        {
            // using stringstream to write matrix
            std::stringstream dataStream(line);
            rawDataMat.conservativeResize(i + 1, numOfCols);
            for (int j = 0; j < numOfCols; j++)
            {
                dataStream >> rawDataMat(i, j);
            }
            i++;
        }
        // close file
        fileToRead.close();
    }
    // error check
    else
    {
        std::cout << "ERROR\nProblem occured while reading given file." << std::endl;
        std::exit(-1);
    }

    // return raw data matrix
    return rawDataMat;
}

//
//
// CALCULATING SUSCEPTIBILITIES (with jackknife samples --> vector form)
//
//

// imZB
auto imZBCalc = [&](Eigen::VectorXd const &imZu, Eigen::VectorXd const &imZs) {
    return (2 * imZu + imZs) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// imZQ
auto imZQCalc = [&](Eigen::VectorXd const &imZu, Eigen::VectorXd const &imZs) {
    return (imZu - imZs) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// imZS
auto imZSCalc = [&](Eigen::VectorXd const &imZs) {
    return -imZs;
};

// ------------------------------------------------------------------------------------------------------------

// ZBB
auto ZBBCalc = [&](Eigen::VectorXd const &Zuu, Eigen::VectorXd const &Zss, Eigen::VectorXd const &Zus, Eigen::VectorXd const &Zud) {
    return (2 * Zuu + Zss + 4 * Zus + 2 * Zud) / 9;
};

// ------------------------------------------------------------------------------------------------------------

// ZQQ
auto ZQQCalc = [&](Eigen::VectorXd const &Zuu, Eigen::VectorXd const &Zss, Eigen::VectorXd const &Zus, Eigen::VectorXd const &Zud) {
    return (5 * Zuu + Zss - 2 * Zus - 4 * Zud) / 9;
};

// ------------------------------------------------------------------------------------------------------------

// ZII
auto ZIIQCalc = [&](Eigen::VectorXd const &Zuu, Eigen::VectorXd const &Zud) {
    return (Zuu - Zud) / 2;
};

// ------------------------------------------------------------------------------------------------------------

// ZBQ
auto ZQBCalc = [&](Eigen::VectorXd const &Zuu, Eigen::VectorXd const &Zss, Eigen::VectorXd const &Zus, Eigen::VectorXd const &Zud) {
    return (Zuu - Zss - Zus + Zud) / 9;
};

// ------------------------------------------------------------------------------------------------------------

// ZBS
auto ZBSCalc = [&](Eigen::VectorXd const &Zss, Eigen::VectorXd const &Zus) {
    return -(Zss + 2 * Zus) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// ZQS
auto ZQSCalc = [&](Eigen::VectorXd const &Zss, Eigen::VectorXd const &Zus) {
    return (Zss - Zus) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// general jackknife error calculator for susceptibilities
auto ZError = [&](Eigen::VectorXd const &Z) {
    // number of jackknife samples
    int N = Z.size() - 2;
    // pre-factor
    double preFactor = (double)(N - 1) / N;
    // estimator / mean
    double estimator = Z.segment(2, N).mean();
    // calculate variance
    double var = 0.;
    for (int i = 0; i < N; i++)
    {
        double val = Z(i + 2) - estimator;
        var += val * val;
    }
    // return error (square root of variance)
    return std::sqrt(preFactor * var);
};
