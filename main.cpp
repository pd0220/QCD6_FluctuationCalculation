// including used header
#include "AnalysisTools.hh"

// ------------------------------------------------------------------------------------------------------------

// main function
// argv[1] --> name of given file with dataset
// argv[2] --> number of jackknife samples
// argv[3] --> number of used susceptibilities
// argv[4] --> which row shall be written to screen
int main(int argc, char **argv)
{
    // check argument list
    if (argc < 5)
    {
        std::cout << "ERROR\nNot enough arguments given." << std::endl;
        std::exit(-1);
    }

    // prepare for reading given file
    // string for file name
    std::string const fileName = argv[1];
    // matrix container for raw data for analysis
    Eigen::MatrixXd const rawDataMat = ReadFile(fileName);

    // number of jackknife samples
    int const jckNum = std::atoi(argv[2]);
    // number of used susceptibilities
    int const ZNum = std::atoi(argv[3]);
    // number of row to be written to screen
    int const rowToWrite = std::atoi(argv[4]);

    // number of cols and rows of raw data matrix
    //int const cols = rawDataMat.cols();
    int const rows = rawDataMat.rows();

    // chemical potentials for baryon numbers and strangeness
    Eigen::VectorXd const muB_T = rawDataMat.col(2);
    Eigen::VectorXd const muS_T = rawDataMat.col(3);

    // susceptibilities (regarding u, d, s flavours) with error and jackknife samples
    // size of vectors
    int const ZSize = 2 + jckNum;
    // vectors to store imZB values and their estimated errors (results)
    Eigen::VectorXd imZBVals(rows);
    Eigen::VectorXd imZBErrs(rows);
    // vectors to store imZQ values and their estimated errors (results)
    Eigen::VectorXd imZQVals(rows);
    Eigen::VectorXd imZQErrs(rows);
    // vectors to store imZS values and their estimated errors (results)
    Eigen::VectorXd imZSVals(rows);
    Eigen::VectorXd imZSErrs(rows);
    // vectors to store ZBB values and their estimated errors (results)
    Eigen::VectorXd ZBBVals(rows);
    Eigen::VectorXd ZBBErrs(rows);
    // vectors to store ZQQ values and their estimated errors (results)
    Eigen::VectorXd ZQQVals(rows);
    Eigen::VectorXd ZQQErrs(rows);
    // vectors to store ZSS values and their estimated errors (results)
    Eigen::VectorXd ZSSVals(rows);
    Eigen::VectorXd ZSSErrs(rows);
    // vectors to store ZBQ values and their estimated errors (results)
    Eigen::VectorXd ZBQVals(rows);
    Eigen::VectorXd ZBQErrs(rows);
    // vectors to store ZBS values and their estimated errors (results)
    Eigen::VectorXd ZBSVals(rows);
    Eigen::VectorXd ZBSErrs(rows);
    // vectors to store ZQS values and their estimated errors (results)
    Eigen::VectorXd ZQSVals(rows);
    Eigen::VectorXd ZQSErrs(rows);
    // vectors to store ZII values and their estimated errors (results)
    Eigen::VectorXd ZIIVals(rows);
    Eigen::VectorXd ZIIErrs(rows);
    // container for "flavour vectors"
    std::vector<Eigen::VectorXd> ZContainer(ZNum);
    // loop for every row
    for (int i = 0; i < rows; i++)
    {
        // filling up vectors
        for (int j = 0; j < ZNum; j++)
        {
            ZContainer[j] = rawDataMat.row(i).segment(4 + j * ZSize, ZSize);
        }
        // calculate (in vector form with jackknife samples)
        Eigen::VectorXd imZB = imZBCalc(ZContainer);
        Eigen::VectorXd imZQ = imZQCalc(ZContainer);
        Eigen::VectorXd imZS = imZSCalc(ZContainer);
        Eigen::VectorXd ZBB = ZBBCalc(ZContainer);
        Eigen::VectorXd ZQQ = ZQQCalc(ZContainer);
        Eigen::VectorXd ZSS = ZSSCalc(ZContainer);
        Eigen::VectorXd ZBQ = ZBQCalc(ZContainer);
        Eigen::VectorXd ZBS = ZBSCalc(ZContainer);
        Eigen::VectorXd ZQS = ZQSCalc(ZContainer);
        Eigen::VectorXd ZII = ZIICalc(ZContainer);
        // save results
        // imZB
        imZBVals(i) = imZB(0);
        imZBErrs(i) = ZError(imZB);
        // imZQ
        imZQVals(i) = imZQ(0);
        imZQErrs(i) = ZError(imZQ);
        // imZS
        imZSVals(i) = imZS(0);
        imZSErrs(i) = ZError(imZS);
        // ZBB
        ZBBVals(i) = ZBB(0);
        ZBBErrs(i) = ZError(ZBB);
        // ZQQ
        ZQQVals(i) = ZQQ(0);
        ZQQErrs(i) = ZError(ZQQ);
        // ZSS
        ZSSVals(i) = ZSS(0);
        ZSSErrs(i) = ZError(ZSS);
        // ZBQ
        ZBQVals(i) = ZBQ(0);
        ZBQErrs(i) = ZError(ZBQ);
        // ZBS
        ZBSVals(i) = ZBS(0);
        ZBSErrs(i) = ZError(ZBS);
        // ZQS
        ZQSVals(i) = ZQS(0);
        ZQSErrs(i) = ZError(ZQS);
        // ZII
        ZIIVals(i) = ZII(0);
        ZIIErrs(i) = ZError(ZII);
    }

    // write results to screen
    std::cout << "imZB = " << imZBVals(rowToWrite) << "    +/-    " << imZBErrs(rowToWrite) << std::endl;
    std::cout << "imZQ = " << imZQVals(rowToWrite) << "    +/-    " << imZQErrs(rowToWrite) << std::endl;
    std::cout << "imZS = " << imZSVals(rowToWrite) << "    +/-    " << imZSErrs(rowToWrite) << std::endl;
    std::cout << "ZBB = " << ZBBVals(rowToWrite) << "    +/-    " << ZBBErrs(rowToWrite) << std::endl;
    std::cout << "ZQQ = " << ZQQVals(rowToWrite) << "    +/-    " << ZQQErrs(rowToWrite) << std::endl;
    std::cout << "ZSS = " << ZSSVals(rowToWrite) << "    +/-    " << ZSSErrs(rowToWrite) << std::endl;
    std::cout << "ZBQ = " << ZBQVals(rowToWrite) << "    +/-    " << ZBQErrs(rowToWrite) << std::endl;
    std::cout << "ZBS = " << ZBSVals(rowToWrite) << "    +/-    " << ZBSErrs(rowToWrite) << std::endl;
    std::cout << "ZQS = " << ZQSVals(rowToWrite) << "    +/-    " << ZQSErrs(rowToWrite) << std::endl;
    std::cout << "ZII = " << ZIIVals(rowToWrite) << "    +/-    " << ZIIErrs(rowToWrite) << std::endl;
}
