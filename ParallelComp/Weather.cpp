
#include "Weather.h"

Weather::Weather() {}
Weather::~Weather() {}

// Load – reads the Brazil temperature file into vectors
// File format (space-separated):
//   StationName  Year  Month  Day  Time  Temperature
//   BERNARDO     1996  12     16   0950  6.0

// Each field is read into a separate vector. All vectors
// share the same index: m_Temp[i] is the temperature of
// reading i, taken in month m_Month[i], year m_Year[i], etc.


bool Weather::Load(std::string FileName)
{   // Initialise variables to hold the data from each line of the file
    std::string place;
	int y, m, d, t; // year, month, day, time
    float tem; // temperature
   
    std::ifstream weatherFile;
    weatherFile.open(FileName);

    if (!weatherFile.is_open())
    {
        std::cerr << "ERROR: Could not open file: " << FileName << std::endl;
        return false;
    }

    std::cout << "Loading: " << FileName << std::endl;

    // Read line by line until end of file
    while (weatherFile >> place >> y >> m >> d >> t >> tem)
    {
        m_Name.push_back(place);
        m_Year.push_back(y);
        m_Month.push_back(m);
        m_Day.push_back(d);
        m_Time.push_back(t);
        m_Temp.push_back(tem);
        m_TempI.push_back((int)(tem * 10.0f)); // integer version for int kernels
    }

    weatherFile.close();
    std::cout << "Loaded " << m_Temp.size() << " readings from "
        << FileName << std::endl << std::endl;
    return true;
}