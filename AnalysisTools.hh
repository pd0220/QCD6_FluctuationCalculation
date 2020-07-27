// functions and methods for jackknife analysis and function fits for correlated data sets

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

// read given file
// expected structure (for this one only):
// x11 | y11 | y_err11 | y_jck11...
// x21 | y21 | y_err21 | y_jck21...
// x12 | y12 | y_err12 | y_jck12...
// x22 | y22 | y_err22 | y_jcj22...
// ... | ... |   ...   |   ...
Eigen::MatrixXd ReadFile(std::string const &fileName)
{
    // start reading
    std::ifstream fileToRead;
    fileToRead.open(fileName);

    // determine number of columns (3 + N_jck)
    std::string firstLine;
    std::getline(fileToRead, firstLine);
    std::stringstream firstLineStream(firstLine);

    // number of columns in given file
    int numOfCols = 0;
    std::string tmpString;
    // count number of writes to temporary string container
    while (firstLineStream >> tmpString)
    {
        numOfCols++;
    }
    fileToRead.close();

    // string for lines
    std::string line;

    // data structure to store data
    // first column is for name
    Eigen::MatrixXd DataMat(0, numOfCols - 1);

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
            DataMat.conservativeResize(i + 1, numOfCols - 1);
            // buffer the name
            std::string tmp;
            dataStream >> tmp;
            for (int j = 0; j < numOfCols - 1; j++)
            {
                dataStream >> DataMat(i, j);
            }
            i++;
        }
        // close file
        fileToRead.close();
    }
    // error check
    else
    {
        std::cout << "Unable to open given file." << std::endl;
        std::exit(-1);
    }

    // return raw data matrix with given structure
    return DataMat;
}

// ------------------------------------------------------------------------------------------------------------

// calculate correlation coefficients of two datasets with given means (faster this way)
double CorrCoeff(Eigen::VectorXd const &one, Eigen::VectorXd const &two, double meanOne, double meanTwo)
{
    // number of jackknife samples
    double NJck = (double)one.size();

    // calculate correlation (not normed)
    double corr = 0;
    for (int i = 0; i < NJck; i++)
    {
        corr += (one(i) - meanOne) * (two(i) - meanTwo);
    }

    // return normed correlation coefficient
    return corr * (NJck - 1) / NJck;
}

// ------------------------------------------------------------------------------------------------------------

// block from the blockdiagonal inverse of covariance matrix
Eigen::MatrixXd BlockCInverse(Eigen::MatrixXd const &rawDataMat, int const &numOfQs, int const &Q)
{
    // jackknife samples to calculate/estimate correlations
    Eigen::MatrixXd JCKs(numOfQs, rawDataMat.cols() - 3);
    for (int i = 0; i < JCKs.cols(); i++)
    {
        for (int j = 0; j < JCKs.rows(); j++)
        {
            JCKs(j, i) = rawDataMat(Q * numOfQs + j, i + 3);
        }
    }

    // get y_err data to compare with correlation results
    Eigen::VectorXd Errs(numOfQs);
    for (int i = 0; i < numOfQs; i++)
    {
        Errs(i) = rawDataMat(Q * numOfQs + i, 2);
    }

    // means to calculate correlations
    std::vector<double> means(numOfQs, 0.);
    for (int i = 0; i < numOfQs; i++)
    {
        means[i] = JCKs.row(i).mean();
    }

    // covariance matrix block
    Eigen::MatrixXd C(numOfQs, numOfQs);
    for (int i = 0; i < numOfQs; i++)
    {
        for (int j = i; j < numOfQs; j++)
        {
            // triangular part
            C(j, i) = CorrCoeff(JCKs.row(i), JCKs.row(j), means[i], means[j]);
            // using symmetris
            if (i != j)
                C(i, j) = C(j, i);
            // compare correlation results with y_err data
            if (i == j && std::abs(Errs(i) * Errs(i) - C(i, j)) / Errs(i) / Errs(i) > 1e-2)
            {
                std::cout << "WARNING\nProblem might occur with covariance matrix." << std::endl;
            }
        }
    }

    // return inverse covariance matrix block
    return C.inverse();
}

// ------------------------------------------------------------------------------------------------------------

// LHS matrix element for given fit (Fourier series)
// ** NOW ** data: imZB --> sine only, ZBB --> cosine only
// * should be more modular *
double LHSMatElement(int const &k, int const &l, Eigen::VectorXd const &xData, std::vector<Eigen::MatrixXd> const &CInvContainer, int const &numOfQs)
{
    // vectors to store base function data --> size is specifically 2
    Eigen::VectorXd baseFunc_k(numOfQs), baseFunc_l(numOfQs);

    // calculate matrix element
    double sum = 0;
    for (int i = 0; i < (int)xData.size() / numOfQs; i++)
    {
        // create vectors
        int index = i * numOfQs;
        baseFunc_k(0) = k * std::sin(k * xData(index));
        baseFunc_l(0) = l * std::sin(l * xData(index));
        baseFunc_k(1) = k * k * std::cos(k * xData(index));
        baseFunc_l(1) = l * l * std::cos(l * xData(index));

        // add to sum the covariance matrix contribution
        sum += baseFunc_l.transpose() * CInvContainer[i] * baseFunc_k;
    }

    // return calculated matrix element
    return sum;
}

// ------------------------------------------------------------------------------------------------------------

// left hand side matrix for linear equation system
Eigen::MatrixXd LHSMat(Eigen::VectorXd const &xData, std::vector<Eigen::MatrixXd> const &CInvContiner, int const &nParams, int const &numOfQs)
{
    // empty (square) matrix with given size
    Eigen::MatrixXd LHS(nParams, nParams);

    // fill matrix
    for (int k = 0; k < nParams; k++)
    {
        for (int l = 1; l < nParams; l++)
        {
            LHS(k, l) = LHSMatElement(k, l, xData, CInvContiner, numOfQs);
        }
    }

    // return LHS matrix
    return LHS;
}

// ------------------------------------------------------------------------------------------------------------

// base functions
double BaseFunc(int const &type, int const &k, double const &x)
{
    // determine type
    if (type == 0)
    {
        return k * std::sin(k * x);
    }
    else if (type == 1)
    {
        return k * k * std::cos(k * x);
    }
    else
    {
        std::cout << "ERROR: overindex in base function." << std::endl;
        std::exit(-1);
    }
}

// ------------------------------------------------------------------------------------------------------------

// RHS vector element for given fit (Fourier series)
double RHSVecElement(int k, Eigen::VectorXd const &yData, Eigen::VectorXd const &xData, std::vector<Eigen::MatrixXd> const &CInvContainer, int const &numOfQs)
{
    // vectors to store base function data --> size is specifically 2
    Eigen::VectorXd baseFunc_k(numOfQs);
    // vector to store given y values --> size is specifically 2
    Eigen::VectorXd yVec(numOfQs);

    // calculate vector element
    double sum = 0;
    for (int i = 0; i < (int)xData.size() / numOfQs; i++)
    {
        // create vectors
        int index = i * numOfQs;
        //baseFunc_k(0) = k * std::sin(k * xData(index));
        //baseFunc_k(1) = k * k * std::cos(k * xData(index));
        for (int j = 0; j < numOfQs; j++)
        {
            baseFunc_k(j) = BaseFunc(j, k, xData(index));
        }
        yVec(0) = yData(index);
        yVec(1) = yData(index + 1);

        // add to sum the covariance matrix contribution
        sum += yVec.transpose() * CInvContainer[i] * baseFunc_k;
    }

    // return calculated matrix element
    return sum;
}

// ------------------------------------------------------------------------------------------------------------

// right hand side vector for linear equation system
Eigen::VectorXd RHSVec(Eigen::VectorXd const &yData, Eigen::VectorXd const &xData, std::vector<Eigen::MatrixXd> const &CInvContainer, int const &nParams, int const &numOfQs)
{
    // empty vector with given size
    Eigen::VectorXd RHS(nParams);

    // fill vector
    for (int k = 0; k < nParams; k++)
    {
        RHS(k) = RHSVecElement(k, yData, xData, CInvContainer, numOfQs);
    }

    // return RHS vector
    return RHS;
}

// ------------------------------------------------------------------------------------------------------------

// error estimation via jackknife method
Eigen::VectorXd JCKErrorEstimation(Eigen::VectorXd const &coeffs, std::vector<Eigen::VectorXd> const &JCKCoeffs)
{
    int length = coeffs.size();
    int N = static_cast<int>(JCKCoeffs.size());
    double preFactor = (double)(N - 1) / N;
    auto sq = [](double x) { return x * x; };

    Eigen::VectorXd sigmaSqVec(length);

    for (int i = 0; i < length; i++)
    {
        sigmaSqVec(i) = 0;
        for (int j = 0; j < N; j++)
        {
            sigmaSqVec(i) += sq(coeffs(i) - JCKCoeffs[j](i));
        }
        sigmaSqVec(i) = std::sqrt(preFactor * sigmaSqVec(i));
    }

    return sigmaSqVec;
}

// ------------------------------------------------------------------------------------------------------------

// chiSquared value (fit quality)
double ChiSq(Eigen::VectorXd const &yData, Eigen::VectorXd const &xData, std::vector<Eigen::MatrixXd> const &CInvContainer, int const &numOfQs, Eigen::VectorXd coeffVector)
{
    // number of fitted paramters
    int nParams = coeffVector.size();
    // sum over blocks
    double sum = 0;
    for (int i = 0; i < (int)xData.size() / numOfQs; i++)
    {
        // delta vector for given block --> size is equal to number of measured quantities
        // y data vector for given block (data point for given x-value)
        Eigen::VectorXd deltaVec(numOfQs), yVec(numOfQs);
        // fill measured data vector and calculate delta vector
        for (int j = 0; j < numOfQs; j++)
        {
            // helper index
            int index = i * numOfQs;
            // fill vector
            yVec(j) = yData(i * numOfQs + j);
            // calculate fitted function for data points
            double deltaSum = 0;
            for (int k = 0; k < nParams; k++)
            {
                deltaSum += coeffVector(k) * BaseFunc(j, k, xData(index));
            }
            // fill delta vector
            deltaVec(j) = yVec(j) - deltaSum;
        }

        // add contribution to chiSquared (matrix multiplication block by block)
        sum += deltaVec.transpose() * CInvContainer[i] * deltaVec;
    }

    // return chiSquared
    return sum;
}

// ------------------------------------------------------------------------------------------------------------

// calculate number of degrees of freedom
int NDoF(Eigen::VectorXd const &xData, Eigen::VectorXd const &coeffVector)
{
    return (int)xData.size() - coeffVector.size() + 1;
}

// ------------------------------------------------------------------------------------------------------------

// AIC weight
double AIC_weight(double const &chiSq, int const &ndof)
{
    return std::exp(-0.5 * (chiSq - 2.0 * ((double)ndof)));
}

// ------------------------------------------------------------------------------------------------------------

// Q weight / test (via GSL libraries)
double Q_weight(double const &chiSq, int const &ndof)
{
    return gsl_cdf_chisq_Q(chiSq, ndof);
}

// ------------------------------------------------------------------------------------------------------------

// calculate original block means (and reducing their number by averaging) from jackknife samples
Eigen::VectorXd JCKReducedBlocks(Eigen::VectorXd const &JCKSamples, int const &divisor)
{
    // number of jackknife samples
    int const lengthOriginal = JCKSamples.size();
    // test if divisor is right
    if ((lengthOriginal % divisor) != 0.)
    {
        std::cout << "Divisor ERROR in Jackknife sample reduction." << std::endl;
        std::exit(-1);
    }

    // empty vector for block values
    Eigen::VectorXd blockVals(lengthOriginal);
    // sum of (original) samples
    double const sum = JCKSamples.sum();
    // calculate block values and add to vector
    for (int i = 0; i < lengthOriginal; i++)
    {
        blockVals[i] = sum - (lengthOriginal - 1) * JCKSamples[i];
    }

    // create new blocks
    // number of new blocks
    int const reduced = lengthOriginal / divisor;
    // vector for new blocks (reduced)
    Eigen::VectorXd newBlocks(divisor);
    // calculate new blocks
    for (int i = 0; i < divisor; i++)
    {
        newBlocks[i] = 0;
        for (int j = 0; j < reduced; j++)
        {
            newBlocks[i] += blockVals[i * reduced + j];
        }
        newBlocks[i] /= reduced;
    }

    // reduced samples
    return newBlocks;
}

// ------------------------------------------------------------------------------------------------------------

// calculate new jackknife samples (after reducing number of blocks)
Eigen::VectorXd JCKSamplesCalculation(Eigen::VectorXd const &Blocks)
{
    // number of new blocks
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
}

// ------------------------------------------------------------------------------------------------------------

// calculate variance for jackknife method
double JCKVariance(Eigen::VectorXd const &JCKSamples)
{
    // size
    int N = JCKSamples.size();
    // mean / estimator
    double estimator = JCKSamples.mean();
    // calculate pre-factor
    double preFactor = (double)(N - 1) / N;
    // calculate sum
    double var = 0.;
    for (int i = 0; i < N; i++)
    {
        double val = JCKSamples[i] - estimator;
        var += val * val;
    }

    // return variance
    return preFactor * var;
}
