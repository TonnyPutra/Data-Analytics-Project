# Air Quality in Shunyi Station

## Setup environment
```
conda create --name base python=3.8
conda activate base
pip install numpy pandas matplotlib seaborn jupyter streamlit
```

## Run streamlit app
```
streamlit run AirQuality-Shunyi.py
```

## Note:
Start by inputing date, the web will process all the data and show the output. <br>
The web is still limited in adjusting the available data, the web can only run with a date input range of 3/1/2013 - 2/28/2017.
