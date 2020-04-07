# Sleep

This package currently contains tools for reading CyberPSG xml annotations, its standardization to achieve a standardized data format, and set of functions working over the standardazided data format performing sleep scoring. 

#### Standardized Data Format

Pandas DataFrame where each row represents a single sleep interval with a given sleep category.
This framework operates with absolute timepoint information stored in a object datetime from a standard python datetime library. These datetime objects are timezone-aware. Therefore, timezone should be specified using tz from dateutil library. 


Pandas DataFrame containing following mandatory fields.

| Key           | Value type    | Values    | Description  |
|:-------------:|:-------------:|:---------:|:------------:|
| annotation    | string        | "REM", "Arrousal", "N1", "N2", "N3", "REM" | Sleep category. |
| start         | datetime      |   - | An absolute timezone-aware timepoint when the given scored sleep epoch begins. |
| end           | datetime      |   - | An absolute timezone-aware timepoint when the given scored sleep epoch ends. |
| duration      | int / float   |   x > 0 | Duration of a given epoch in seconds. |



## Data
Functions employed in the Analysis package for parsing and standardization of CyberPSG annotations. Other formats will be possibly added in future.


## Analysis
Complete tools for reading CyberPSG annotations and consequent sleep scoring. Sleep scoring functions can be utilized with any sleep data in the standardized data format specified above. Please, see an example bellow.


### Example 
```python
import sys
sys.path.append('D:\\MayoWork\\MSEL\\lib') # Set the source path to the lib folder of this python package.


from Sleep.Analysis import SleepAnnotationsScoring, score_night, print_score

path = r'D:\MayoWork\MSEL\ExampleData\CyberPSGAnnotation.xml'
Annotations = SleepAnnotationsScoring(path)
for k in range(Annotations.days):
    day = Annotations.get_day(k)
    score = score_night(day, plot=True)
    print('#############################################')
    print('Day {0}'.format(k))
    print_score(score)
```
