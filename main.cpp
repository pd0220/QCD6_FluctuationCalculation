// including used header
#include "AnalysisTools.hh"

// ------------------------------------------------------------------------------------------------------------

// main function
// argv[1] --> name of given file with dataset
// argv[2] --> number of jackknife samples
// argv[3] --> number of used susceptibilities
int main(int argc, char **argv)
{
    // check argument list
    if (argc < 4)
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
    int const ZNum = std::atoi(argv[3]);

    // number of cols and rows of raw data matrix
    int const cols = rawDataMat.cols();
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
    // vectors to store imZB values and their estimated errors (results)
    Eigen::VectorXd ZBBVals(rows);
    Eigen::VectorXd ZBBErrs(rows);
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
        // labeling and saving susceptibilities
        Eigen::VectorXd imZu = ZContainer[0];
        Eigen::VectorXd imZs = ZContainer[1];
        Eigen::VectorXd Zuu = ZContainer[2];
        Eigen::VectorXd Zud = ZContainer[3];
        Eigen::VectorXd Zus = ZContainer[4];
        Eigen::VectorXd Zss = ZContainer[5];
        // calculate imZB and ZBB (in vector form with jackknife samples)
        Eigen::VectorXd imZB = imZBCalc(imZu, imZs);
        Eigen::VectorXd ZBB = ZBBCalc(Zuu, Zss, Zus, Zus);
        // save results
        // imZB
        imZBVals(i) = imZB(0);
        imZBErrs(i) = ZError(imZB);
        // ZBB
        ZBBVals(i) = ZBB(0);
        ZBBErrs(i) = ZError(ZBB);
    }

    // write to screen (imZB)
    for (int i = 0; i < imZBVals.size(); i++)
    {
        std::cout << imZBVals(i) << "    +/-    " << imZBErrs(i) << std::endl;
    }
}
