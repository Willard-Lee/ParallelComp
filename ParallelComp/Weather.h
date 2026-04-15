#pragma once

// Reads the Brazil temperature dataset into separate vectors.
// File format per line:
//   StationName  Year  Month  Day  Time  Temperature
//   BERNARDO     1996  12     16   0950  6.0

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

class Weather
{
public:
    Weather();
    ~Weather();

    // Load the data file – returns true if successful
    bool Load(std::string FileName);

    // Getters – all vectors are the same length; index i is one reading
    std::vector<float>& GetTemp() { return m_Temp; }
    std::vector<int>& GetTempI() { return m_TempI; }
    std::vector<int>& GetDay() { return m_Day; }
    std::vector<int>& GetYear() { return m_Year; }
    std::vector<int>& GetMonth() { return m_Month; }
    std::vector<int>& GetTime() { return m_Time; }
    std::vector<std::string>& GetName() { return m_Name; }

private:
    std::vector<float>       m_Temp;   // temperature as float
    std::vector<int>         m_TempI;  // temperature * 10 as int (for integer kernels)
    std::vector<int>         m_Day;
    std::vector<int>         m_Year;
    std::vector<int>         m_Month;
    std::vector<int>         m_Time;
    std::vector<std::string> m_Name;   // station name
};