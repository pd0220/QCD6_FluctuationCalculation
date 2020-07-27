// including used header
#include "AnalysisTools.hh"

// ------------------------------------------------------------------------------------------------------------

// main function
// argv[1] --> name of given file with dataset
//
//
int main(int argc, char **argv)
{
    // check argument list
    if (argc < 2)
    {
        std::cout << "ERROR\nNot enough arguments given." << std::endl;
        std::exit(-1);
    }

    // prepare for reading given file
    // string for file name
    std::string fileName = argv[1];
    // matrix container for raw data for analysis
    Eigen::MatrixXd rawDataMat = ReadFile(fileName);
}
