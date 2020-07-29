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
// labeling
// imZu --> ZContainer[0];
// imZs --> ZContainer[1];
// Zuu  --> ZContainer[2];
// Zud  --> ZContainer[3];
// Zus  --> ZContainer[4];
// Zss  --> ZContainer[5];
//
//

// imZB
auto imZBCalc = [&](std::vector<Eigen::VectorXd> const &Z) {
    return (2 * Z[0] + Z[1]) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// imZQ
auto imZQCalc = [&](std::vector<Eigen::VectorXd> const &Z) {
    return (Z[0] - Z[1]) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// imZS
auto imZSCalc = [&](std::vector<Eigen::VectorXd> const &Z) {
    return -Z[1];
};

// ------------------------------------------------------------------------------------------------------------

// ZBB
auto ZBBCalc = [&](std::vector<Eigen::VectorXd> const &Z) {
    return (2 * Z[2] + Z[5] + 4 * Z[4] + 2 * Z[3]) / 9;
};

// ------------------------------------------------------------------------------------------------------------

// ZQQ
auto ZQQCalc = [&](std::vector<Eigen::VectorXd> const &Z) {
    return (5 * Z[2] + Z[5] - 2 * Z[4] - 4 * Z[3]) / 9;
};

// ------------------------------------------------------------------------------------------------------------

// ZSS
auto ZSSCalc = [&](std::vector<Eigen::VectorXd> const &Z) {
    return Z[5];
};

// ------------------------------------------------------------------------------------------------------------

// ZII
auto ZIICalc = [&](std::vector<Eigen::VectorXd> const &Z) {
    return (Z[2] - Z[3]) / 2;
};

// ------------------------------------------------------------------------------------------------------------

// ZBQ
auto ZBQCalc = [&](std::vector<Eigen::VectorXd> const &Z) {
    return (Z[2] - Z[5] - Z[4] + Z[3]) / 9;
};

// ------------------------------------------------------------------------------------------------------------

// ZBS
auto ZBSCalc = [&](std::vector<Eigen::VectorXd> const &Z) {
    return -(Z[5] + 2 * Z[4]) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// ZQS
auto ZQSCalc = [&](std::vector<Eigen::VectorXd> const &Z) {
    return (Z[5] - Z[4]) / 3;
};

// ------------------------------------------------------------------------------------------------------------

// calculate variance (for jackknife samples)
auto JCKVariance = [&](Eigen::VectorXd const &JCKSamples) {
    // size of vector
    int N = JCKSamples.size();
    // pre-factor
    double preFactor = (double)(N - 1) / N;
    // estimator / mean
    double estimator = JCKSamples.mean();
    // calculate variance
    double var = 0.;
    for (int i = 0; i < N; i++)
    {
        double val = JCKSamples(i) - estimator;
        var += val * val;
    }
    // return variance
    return preFactor * var;
};

// ------------------------------------------------------------------------------------------------------------

// general jackknife error calculator for susceptibilities
auto ZError = [&](Eigen::VectorXd const &Z) {
    return std::sqrt(JCKVariance(Z.segment(2, Z.size() - 2)));
};

// ------------------------------------------------------------------------------------------------------------

// calculate original block means (and reducing their number by averaging) from jackknife samples
auto JCKReducedBlocks = [&](Eigen::VectorXd const &JCKSamplesOld, int const &divisor) {
    // number of samples
    int const NOld = JCKSamplesOld.size();
    // test if divisor is correct for the original sample number
    if ((NOld % divisor) != 0.)
    {
        std::cout << "ERROR\nIncorrect divisor during Jackknife sample reduction." << std::endl;
        std::exit(-1);
    }
    // empty vector for block values
    Eigen::VectorXd blockVals(NOld);
    // sum of (original) samples
    double const sum = JCKSamplesOld.sum();
    // calculate block values and add to vector
    for (int i = 0; i < NOld; i++)
    {
        blockVals(i) = sum - (NOld - 1) * JCKSamplesOld(i);
    }
    // create new blocks
    // old blocks to add up for new blocks
    int const reduced = NOld / divisor;
    // vector for new blocks (reduced)
    Eigen::VectorXd newBlocks(divisor);
    // calculate new blocks
    for (int i = 0; i < divisor; i++)
    {
        newBlocks(i) = 0;
        for (int j = 0; j < reduced; j++)
        {
            newBlocks(i) += blockVals(i * reduced + j);
        }
        newBlocks(i) /= reduced;
    }
    // return new blocks
    return newBlocks;
};

// ------------------------------------------------------------------------------------------------------------

// calculate jackknife samples from block means
auto JCKSamplesCalculation = [&](Eigen::VectorXd const &Blocks) {
    // number of blocks
    int const lengthBlocks = Blocks.size();
    // vector for jackknife samples
    Eigen::VectorXd Samples(lengthBlocks);
    // copy data to std::vector
    std::vector<double> tempVec(Blocks.data(), Blocks.data() + lengthBlocks);
    // create jackknife samples
    for (int i = 0; i < lengthBlocks; i++)
    {
        // copy data
        std::vector<double> tempJCKVec = tempVec;
        // delete ith element
        tempJCKVec.erase(tempJCKVec.begin() + i);
        // calculate mean
        Samples[i] = std::accumulate(tempJCKVec.begin(), tempJCKVec.end(), 0.) / (lengthBlocks - 1);
    }
    // return new jackknife samples
    return Samples;
};

// ------------------------------------------------------------------------------------------------------------

// general jackknife error calculator for susceptibilities with sample number reductions (according to divisors)
auto ZErrorJCKReduced = [&](Eigen::VectorXd const &Z, int const &divisor) {
    // number of jackknife samples
    int NOld = Z.size() - 2;
    // get new jackknife samples via calculating old blocks and reducing their number by averaging
    Eigen::VectorXd JCKSamples = JCKSamplesCalculation(JCKReducedBlocks(Z.segment(2, NOld), divisor));
    // return jackknfife error
    return std::sqrt(JCKVariance(JCKSamples));
};
