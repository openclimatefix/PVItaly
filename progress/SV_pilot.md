# SV Pilot

## Summary

We forecasted PV production over the next 4 hours for 4 sites in Italy. The average error was XXX. 

## Data

#### Sites

We are forecasting for four sites across Italy. 

|    | Location | Capacity [kW]
| ----------- | ----------- | --- |
| 1      | Belluno       | 35
| 2   | Bari        | 25
| 3   | Bari        | 20
| 4   | Venice        | 20 

#### SV specific filtering

The following filters were used.

1. We removed any negative value
2. Remove any values above 10^6
3. Remove any values above mean + 5 stds. This gets rid of any anomalies
4. Remove specific datetimes for site 4 (Universita Ca Foscari), 2021-06-09 to 2021-07-12

### PVoutput.org

We downloaded 436 sites from Italy from [PVoutput.org](https://pvoutput.org/region.jsp?country=117). 

![image](./PV_sites.png)

### NWP 

### General

We add the sun's azimuth and elevation angle as a model input. 
This is highly predictable and clearly has strong correlation with pv production 

We also normalized the data by the capacity value. This normalizes the pv production, so all values are between 0 and 1. 

## Models

### Baseline

It is always good to baseline the models with some very simple models, 
in order to get an understanding of the statistics. 

The following different baselines were looked at:
- Zero: Always predict zero 
- Persist: Use the last values for the forecast of the 

See results [here](SV_pilot.md#Results).  




###  Fully Connected Neural Network

The idea is to use a 3 hidden layered of full connection neural network. 
This takes all the inputs above, passes them through the layers and 
then produces predictions at 15 mins intervals for the next 4 hours.

## Training

We divided the data into 2021 and 2022. 
We trained our models using 2022, and then validated our results on 2021.

Training our models took approxiately X hours

## Results

|               | MAE [%]   | MSE [%]
| -----------   | --------- | --- 
| Zero          | 10.4      | 4.7
| Persist       | 7.8       | 2.4
| SV sites only | 3.25      | 0.475
| All sites     |           | 

## Next Steps

### Data Satellite

It would be great to use Satellite information aswell and our models could easily be extended in the future.
In previous models with have seen 16% in accuracy.  

### Models

We would like to use some deep neural networks to enchance our results. 

1. PVnet - [link](TODO)
2. Metnet - [link](TODO)

Both these methods would allow a lot more spatial data to be used to make the predictions more accuracy. 
