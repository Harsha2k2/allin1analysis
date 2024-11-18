# README for Advanced Data Analyzer
## Overview
The Advanced Data Analyzer is a Python tool designed to perform comprehensive data analysis on pandas DataFrames. It provides functionalities to identify missing values, handle duplicates, visualize data distributions, and generate detailed reports in various formats, including Word documents.
## Features
#### Identify Missing Values: Detects columns with missing values and provides statistics.
#### Categorize Columns: Classifies columns into numeric, categorical, datetime, and boolean types.
#### Handle Duplicates: Identifies duplicate rows and offers the option to remove them.
#### Handle Constant Columns: Detects columns with constant values and can remove them if desired.
#### Generate Visualizations: Creates visual representations of the data, including boxplots and distribution plots.
#### Export Reports: Generates a comprehensive data analysis report in a Word document format.
# Installation
To use the Advanced Data Analyzer, ensure you have Python installed along with the required libraries. You can install the necessary libraries using pip:

```
pip install pandas numpy matplotlib seaborn xlsxwriter jinja2 pdfkit python-docx
```

Usage
Import the Class:
Import the AdvancedDataAnalyzer class from your script.
python
from advanced_data_analyzer import AdvancedDataAnalyzer

Create a DataFrame:
You can either load your own DataFrame or use the provided sample data generator.
python
import pandas as pd

# Generate sample data for testing
df = generate_sample_data()

Initialize the Analyzer:
Create an instance of the AdvancedDataAnalyzer class with your DataFrame.
python
analyzer = AdvancedDataAnalyzer(df)

Perform Analysis:
Call methods to analyze your data:
python
# Find missing values
missing_values = analyzer.find_missing_values()

# Handle duplicates
duplicates = analyzer.handle_duplicates(remove=True)

# Generate visualizations
visualizations = analyzer.generate_visualizations()

# Export to Word document
```
analyzer.export_to_word('data_analysis_report.docx')
```
Run the Script:
If you want to run the script directly, include the following in your main execution block:
python
```
if __name__ == "__main__":
    df = generate_sample_data()  # Or load your own DataFrame
    analyzer = AdvancedDataAnalyzer(df)
    analyzer.export_to_word('data_analysis_report.docx')
    print("Data analysis report saved to data_analysis_report.docx")
```
# Visualizations

### Boxplot
![Boxplot of Numeric Columns](https://github.com/Harsha2k2/allin1analysis/blob/main/Boxplog.png)
### Distribution
![Distribution of Data](https://github.com/Harsha2k2/allin1analysis/blob/main/Distributions.png)
# Example Output
The tool generates a Word document named data_analysis_report.docx, which includes:
A summary of missing values.
Visualizations such as boxplots and distribution plots.
